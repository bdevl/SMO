import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Union, List
import numpy as np
import torch
from mpi4py.futures import MPIPoolExecutor
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import norm as normal_scipy
from tqdm import tqdm
from SMO.discriminative import DiscriminativeCNN
from SMO.doe import GaussianDesignOfExperiment
from SMO.doe import SingularPoint
from SMO.microstructure.homogenization import (
    PoissonHomogenizer,
    PeriodicHomogenizer,
    CombinedHomogenizer,
)
from SMO.microstructure.randomfields import (
    DifferentiableGaussianRandomField,
    GaussianGridKernel,
    UnboundedRandomFieldParameters,
    SoftmaxRandomFieldParameters,
)
from SMO.optimization.inference import LowRankVariationalInference
from SMO.optimization.objectives import (
    IntervalObjective,
    KullbackLeiblerObjective,
    GaussianMultivariateTargetDistribution,
)
from SMO.utils import (
    BinarizeDataTransform,
    HyperbolicDataTransform,
    Gaussian2D,
    plotdensity2d,
    substitute_defaults,
)
from genutils import backend
from genutils import ensure_folder
from parallel.utils import DummyProcessPool
from utils.database import PoolTracker


class CaseFactory(object):
    def __init__(self, args, folder):

        assert isinstance(args, dict)
        self._args = args

        if "volumefraction" not in self._args:
            self._args["volumefraction"] = 0.5

        self._pdim = None

        self._folder_base = ensure_folder(folder)

        self._cutoff = None

    @property
    def pdim(self):

        if self._pdim is None:
            rf_, _ = self.rf()
            self._pdim = rf_.kernel.pdim
            assert isinstance(self._pdim, int)
            assert self._pdim > 0

        return self._pdim

    @property
    def _folder(self):

        return self._folder_base + "cr{}/".format(self.cr)

    @property
    def nx(self):
        assert "nx" in self._args, "nx has not been set in self._args dict"
        return self._args["nx"]

    @property
    def _target_sel(self):
        raise DeprecationWarning("user target, instead of _target_sel")

    @property
    def target(self):
        assert "target" in self._args, "target is not defined in self._args dict"
        return self._args["target"]

    @property
    def _N_training(self):
        raise DeprecationWarning

    @property
    def _N_validation(self):
        raise DeprecationWarning

    def check(self):

        self._check_hmg()

    def _check_hmg(self):

        hmg = self.hmg()
        admissible = hmg.admissible
        for sel in self.target:
            assert (
                sel in admissible
            ), "A check revealed that the homogenized property ({}) cannot be computed by {}. Valid properties are: {}".format(
                sel, hmg.__class__.__name__, admissible
            )

    def load_model(self, model):

        try:
            path = self._folder + "model/trained.pt"
            model.load(path)
        except Exception as e:
            raise e

        return model

    def save_model(self, model):

        os.makedirs(self._folder + "model", exist_ok=True)
        path = self._folder + "model/trained.pt"
        model.save(path)

    @property
    def Nw(self):
        raise DeprecationWarning

    @property
    def w_max(self):
        raise DeprecationWarning

    @property
    def phase_high(self):

        assert "cr" in self._args, "Contrast ratio has not been set"
        return self._args["cr"]

    @property
    def phase_low(self):

        return 1

    @property
    def cutoff(self):

        if self._cutoff is None:
            assert (
                "volumefraction" in self._args
                and isinstance(self._args["volumefraction"], (float))
                and 0.1 <= self._args["volumefraction"] <= 0.9
            )
            cutoff = -normal_scipy.ppf(self._args["volumefraction"])
            assert -1.3 <= cutoff <= 1.3
            if self._args["volumefraction"] == 0.5:
                assert cutoff == 0
            self._cutoff = cutoff

        return self._cutoff

    @property
    def cuda_available(self):
        return torch.cuda.is_available()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def device(self):
        return torch.device("cuda:0") if self.cuda_available else torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    @property
    def cr(self):
        return self._args["cr"]

    def get_datafactory(self, *args, **kwargs):

        return DataFactory(self.__class__.__name__, self._args, *args, **kwargs)

    def _convert_kappa_to_tensor(self, kappa: dict) -> torch.Tensor:

        for sel in self.target:
            assert (
                sel in kappa
            ), "The desired homogenized property ({}) has not been computed. Valid keys: {}".format(
                sel, kappa.keys()
            )

        assert all([kappa[sel].dim() == 1 for sel in self.target])
        return torch.stack([kappa[sel] for sel in self.target], dim=1)

    @classmethod
    def FromIdentifier(
        cls, identifier: str, *args, return_class_not_instance=False, **kwargs
    ):

        classname = identifier
        try:
            factory_class = globals()[classname]
        except KeyError:
            raise KeyError(
                'The factory with identifier "{}" is not defined.'.format(identifier)
            )

        if return_class_not_instance:
            return factory_class

        factory = factory_class(*args, **kwargs)
        return factory

    def objfct_ref(self, interval, return_info=False):

        raise DeprecationWarning("Implementation has to be updated.")

    def plot_marginal_distribution_of_phi(
        self, phi: np.ndarray, N: int, nbins: int = 50
    ):

        kappa = self.kappa_from_phi(phi, N).detach().cpu().numpy()
        assert kappa.shape[1] == 2, "assumes 2D"

        k1 = kappa[:, 0]
        k2 = kappa[:, 1]

        plotdensity2d(kappa[:, 0], kappa[:, 1], nbins=nbins)

        return k1, k2

    @torch.no_grad()
    def kappa_from_phi(self, phi, N, return_X_cnn=False) -> torch.Tensor:

        doe = SingularPoint(phi, N=N)
        datafct_gnr = self.get_datafactory(num_samples_per_phi=1, DOE=doe)

        RES = datafct_gnr(retain_X_g=return_X_cnn)
        kappa_keys = RES["kappa"][0][0].keys()
        kappa = dict()
        for kappa_key in kappa_keys:
            kappa[kappa_key] = torch.zeros(len(doe))

        for n in range(len(doe)):

            for j, mkey in enumerate(kappa_keys):
                kappa[mkey][n] = RES["kappa"][n][0][mkey]

        kappa = self._convert_kappa_to_tensor(kappa)

        if not return_X_cnn:
            return kappa
        else:
            dtransform = self.dtransform()
            X_g = torch.stack([RES["X_g"][n][0] for n in range(len(RES["X_g"]))])
            assert X_g.ndim == 3
            assert X_g.shape[0] == kappa.shape[0]
            X_cnn = dtransform(X_g)
            return kappa, X_cnn

    @property
    def N_training_available(self):

        return self._args["N_training"]

    def data(
        self,
        N_training=None,
        N_validation=None,
        ForceRecompute=False,
        mpi_threads=None,
        shared_memory_parallelization=False,
        permute_returned_data=True,
        permutation_tr=None,
    ):

        try:
            if ForceRecompute:
                raise Exception

            print(
                "Attempting to load data from ... : {}".format(self._folder + "data/")
            )

            data_train = torch.load(self._folder + "data/train.pt")
            data_val = torch.load(self._folder + "data/val.pt")
        except:

            doe_training, doe_validation = self.doe()
            datafct_gnr_tr = self.get_datafactory(
                num_samples_per_phi=1, DOE=doe_training
            )
            datafct_gnr_val = self.get_datafactory(
                num_samples_per_phi=1, DOE=doe_validation
            )

            data_train = datafct_gnr_tr.training_data(
                mpi_num_workers=mpi_threads,
                shared_memory_parallelization=shared_memory_parallelization,
            )
            data_val = datafct_gnr_val.training_data(
                mpi_num_workers=mpi_threads, shared_memory_parallelization=False
            )
            os.makedirs(self._folder + "data", exist_ok=True)
            torch.save(data_train, self._folder + "data/train.pt")
            torch.save(data_val, self._folder + "data/val.pt")

        kappa_tr, X_g_tr, X_b_tr, phi_tr = data_train
        kappa_vl, X_g_vl, X_b_vl, phi_vl = data_val

        kappa_tr = self._convert_kappa_to_tensor(kappa_tr)
        kappa_vl = self._convert_kappa_to_tensor(kappa_vl)

        if permute_returned_data:

            if permutation_tr is not None:
                assert (
                    isinstance(permutation_tr, torch.Tensor)
                    and permutation_tr.ndim == 1
                    and len(permutation_tr) == kappa_tr.shape[0]
                ), "Permutation tensor passed to data() not proper."
                indperm_tr = permutation_tr
            else:
                indperm_tr = torch.randperm(kappa_tr.shape[0])

            indperm_vl = torch.randperm(kappa_vl.shape[0])

            kappa_tr = kappa_tr[indperm_tr]
            kappa_vl = kappa_vl[indperm_vl]

            X_g_tr = X_g_tr[indperm_tr]
            X_b_tr = X_b_tr[indperm_tr]
            phi_tr = phi_tr[indperm_tr]

            X_g_vl = X_g_vl[indperm_vl]
            X_b_vl = X_b_vl[indperm_vl]
            phi_vl = phi_vl[indperm_vl]

        if N_training is not None:
            assert N_training <= kappa_tr.shape[0]
            kappa_tr = kappa_tr[0:N_training]
            X_g_tr = X_g_tr[0:N_training]
            X_b_tr = X_b_tr[0:N_training]
            phi_tr = phi_tr[0:N_training]

        if N_validation is not None:
            assert N_validation <= kappa_vl.shape[0]
            kappa_vl = kappa_vl[0:N_validation]
            X_g_vl = X_g_vl[0:N_validation]
            X_b_vl = X_b_vl[0:N_validation]
            phi_vl = phi_vl[0:N_validation]

        data_tr = dict()
        data_tr["kappa"] = kappa_tr
        data_tr["X_g"] = X_g_tr
        data_tr["X_b"] = X_b_tr
        data_tr["phi"] = phi_tr

        data_val = dict()
        data_val["kappa"] = kappa_vl
        data_val["X_g"] = X_g_vl
        data_val["X_b"] = X_b_vl
        data_val["phi"] = phi_vl

        return data_tr, data_val

    def interpolator(self, objective, rfp_min=None, rfp_max=None):

        assert hasattr(
            objective, "assess"
        ), 'objective needs to provide "assess" interface'

        phi_ref, kappas = self.reference_kappas()
        assert kappas.shape[2] == 2 and kappas.ndim == 3

        objfct_ref = list()

        for n in range(len(phi_ref)):
            objfct_ref.append(objective.assess(kappas[n]))

        doe = self.doe_reference()
        objfct_ref = np.array(objfct_ref)

        interpolator = LinearNDInterpolator(np.vstack([m for m in doe]), objfct_ref)

        def my_interpolator(phival: Union[np.ndarray, List[float]]) -> float:

            if isinstance(phival, list):
                phival = np.array(phival)
            assert isinstance(
                phival, np.ndarray
            ), "type of phi is not np.ndarray, but {}".format(type(phival))
            assert phival.ndim == 1, "Shape of phi is {} (needs to be 1D)".format(
                phival.shape
            )
            assert len(phival)

            hd = 1e-6
            for ind in range(len(phival)):
                if rfp_min is not None:
                    assert isinstance(rfp_min, np.ndarray)
                    if phival[ind] <= rfp_min[ind]:
                        phival[ind] = rfp_min[ind] + hd
                if rfp_max is not None:
                    assert isinstance(rfp_max, np.ndarray)
                    if phival[ind] >= rfp_max[ind]:
                        phival[ind] = rfp_max[ind] - hd

            val = interpolator(phival)

            if isinstance(val, np.ndarray):
                assert len(val) == 1 and val.ndim == 1
                val = val.item()

            if isinstance(val, list):
                assert len(val) == 1
                val = val[0]

            assert isinstance(val, float) and not np.isnan(
                val
            ), "Interpolator decided to return a non-float (or NaN) value : val = {}".format(
                val
            )
            return val

        return my_interpolator

    def reference_microstructures(
        self, N_samples=2048, ForceRecompute=False, ShowProgress=False, return_doe=False
    ):

        doe = self.doe_reference()

        try:
            if ForceRecompute:
                raise Exception
            data_doe_samples = torch.load(self._folder + "reference/rf_samples.pt")
        except:

            transformer = self.dtransform()
            rf, _ = self.rf()

            if self.cuda_available:
                rf = rf.to(device=torch.device("cuda:0"))

            X_global = list()

            numele = len(doe) * N_samples * np.prod(rf.ns)
            storage_in_mb = (numele * 4) / (1024 ** 2)

            if ShowProgress:
                doe_iterator = tqdm(doe)
            else:
                doe_iterator = doe

            with torch.no_grad():
                for i, phi in enumerate(doe_iterator):

                    rf.set_phi(phi)
                    X_g = rf.rsample_batch(batch_size=N_samples)
                    X_g.unsqueeze_(1)
                    X_global.append(transformer(X_g).to(device=torch.device("cpu")))

            os.makedirs(self._folder + "reference", exist_ok=True)
            torch.save(X_global, self._folder + "reference/rf_samples.pt")
            data_doe_samples = X_global

        if return_doe:
            return data_doe_samples, doe
        else:
            return data_doe_samples

    def dtransform(self):

        if self._args["binarize"]:
            return BinarizeDataTransform(
                self._args["nx"],
                cutoff=self.cutoff,
                phase_low=-1,
                phase_high=1,
                image_channel=True,
            )
        else:
            return HyperbolicDataTransform(
                nx=self._args["nx"], eps=self._args["eps"], cutoff=self.cutoff
            )

    def htransform(self):

        return BinarizeDataTransform(
            self._args["nx"],
            cutoff=self.cutoff,
            phase_low=self.phase_low,
            phase_high=self.phase_high,
            image_channel=False,
        )

    def rf(self):
        raise NotImplementedError

    def hmg(self, constrain_to_target=False):

        return self._hmg()

    def _hmg(self):
        raise NotImplementedError

    def discriminative(self, *args, **kwargs):
        raise NotImplementedError


class ChannelizedFlow32(CaseFactory):
    def __init__(self, args, path="hdata/ChannelizedFlow32/", called_by_subclass=False):

        super().__init__(args, path)

        if called_by_subclass:
            return

        default_values = {
            "nx": 32,
            "cr": 50,
            "N_training": 4096,
            "N_validation": 1024,
            "binarize": False,
            "eps": 25,
            "target": ["xx", "yy"],
            "N_g": 4,
            "sigma_w": 10,
            "N_w": 25,
            "w_max": 65.0,
        }

        substitute_defaults(default_values, self._args)

        self._N_reference_phi_points = 50
        assert self._args["nx"] == 32

    def phi_target(self, return_index=False):

        phi_target = -5 * np.ones(self.pdim)
        phi_target[-self._args["N_g"]] = 5

        if return_index:
            ind = np.argmax(phi_target)
            return phi_target, ind

        return phi_target

    def _hmg(self):

        return PoissonHomogenizer.FromImageResolution(self._args["nx"])

    def doe(self, type=None):

        doe_training = GaussianDesignOfExperiment(
            dim=self.pdim, N=self._args["N_training"]
        )
        doe_validation = GaussianDesignOfExperiment(
            dim=self.pdim, N=self._args["N_validation"]
        )

        if type is None:
            return doe_training, doe_validation
        elif type.lower() == "training":
            return doe_training
        elif type.lower() == "validation":
            return doe_validation

    def doe_reference(self):

        return GaussianDesignOfExperiment(dim=self.pdim, N=self._N_reference_phi_points)

    def rf(self, force_cpu=True):

        nx = self._args["nx"]

        if force_cpu:
            device = torch.device("cpu")
        else:
            device = self.device

        kernel = GaussianGridKernel(
            2,
            self._args["w_max"],
            self._args["N_w"],
            self._args["N_g"],
            sigma_w=self._args["sigma_w"],
            init_uniform=False,
        )
        kernel = kernel.to(dtype=self.dtype, device=device)
        rf = DifferentiableGaussianRandomField(
            nx, kernel, 2, dtype=self.dtype, device=device
        )
        rf = rf.to(dtype=self.dtype, device=device)
        rfp = UnboundedRandomFieldParameters(kernel)

        return rf, rfp

    def discriminative(self, trained=False):

        surrogate = DiscriminativeCNN(
            nx=32,
            filters=[4, 8, 16],
            dim_out=len(self.target),
            dropout=0.05,
            num_additional_full_layers=1,
            make_deterministic=False,
            gate=False,
        )

        if trained:
            surrogate.load(self._folder + "/surrogate/trained.pt")

        surrogate.eval()

        return surrogate

    def objective(self, wmodel, inference, dtype, device, type="KLD"):

        raise DeprecationWarning

    @classmethod
    def default_settings(cls, DEBUG=False):

        if DEBUG:
            N_data_acquisitions = 1
            M_steps = 2
            N_em_max = 5
            N_epochs = 20
            N_obj_fct = 64
            N_batches = 10
        else:
            N_data_acquisitions = 6
            M_steps = 2
            N_em_max = 350
            N_epochs = 3000
            N_obj_fct = 512
            N_batches = 20000

        fargs = dict()

        cargs = dict()
        cargs["N_monte_carlo_elbo"] = 256
        cargs["N_em_max_steps"] = N_em_max
        cargs["N_objective_fct_monte_carlo"] = N_obj_fct
        cargs["patience"] = 15

        cargs["N_training_init"] = 512
        cargs["N_add"] = 512
        cargs["N_candidates"] = 4096
        cargs["N_data_acquisitions"] = N_data_acquisitions
        cargs["N_training_baseline"] = [512, 2048, 4096]
        cargs["N_validation"] = 256

        cargs["cooldown"] = 100  # 15
        cargs["active_strategy"] = "kldlogscore"
        cargs["M_steps"] = M_steps

        cargs["VI_SGD_iterations"] = 85
        cargs["VI_lowrank_dim"] = 40
        cargs["VI_SGD_monte_carlo"] = 10
        cargs["VI_em_samples"] = 128 + 64

        targs = dict()
        targs["N_epochs"] = None
        targs["N_batches"] = N_batches
        targs["batch_size"] = 128
        targs["lr"] = 1e-3
        targs["weight_decay"] = 1e-5
        targs["verbose"] = True
        targs["reset_model"] = True

        return fargs, cargs, targs


class ChannelizedFlow64(ChannelizedFlow32):
    def __init__(self, args, path="hdata/ChannelizedFlow64/"):

        super().__init__(args, path, called_by_subclass=True)

        default_values = {
            "nx": 64,
            "cr": 50,
            "N_training": 4 * 4096,
            "N_validation": 1024,
            "binarize": False,
            "eps": 25,
            "target": ["xx", "yy"],
            "N_g": 10,
            "sigma_w": 12,
            "N_w": 35,
            "w_max": 65.0,
            "lr_m_step": 0.25,
            "volumefraction": 0.50,
        }

        substitute_defaults(default_values, self._args)

    def objective(
        self, wmodel, cargs: dict, dtype, device, type="KLD", only_mean=False
    ):

        assert isinstance(cargs, dict)

        if type.lower() == "bound":
            bounds = np.array([[21.0, 25.0], [2.5, 3.5]])
            inference = LowRankVariationalInference(
                cargs["VI_SGD_iterations"],
                cargs["VI_lowrank_dim"],
                cargs["VI_SGD_monte_carlo"],
                N_samples_em=cargs["VI_em_samples"],
            )
            objective = IntervalObjective.FromBounds(
                wmodel, inference, bounds, dtype=dtype, device=device
            )

        elif type.lower() == "kld":

            N_samples = 20
            mean = np.array([20.5, 3.5])
            cov = np.array([[0.60, -0.03], [-0.03, 0.01]])

            q = Gaussian2D(mean, cov)
            kappas = q.sample(N_samples)
            kappas_t = torch.tensor(kappas, dtype=dtype, device=device)

            if N_samples == 1 and not only_mean:
                kappas_t = kappas_t.view(1, -1)

            if only_mean:
                kappas_t = torch.tensor(mean, dtype=dtype, device=device).view(1, -1)
                N_samples = 1

            N_updates = None

            inference = LowRankVariationalInference(
                cargs["VI_SGD_iterations"],
                cargs["VI_lowrank_dim"],
                cargs["VI_SGD_monte_carlo"],
                batch_size=N_samples,
                N_samples_em=cargs["VI_em_samples"],
            )

            objective = KullbackLeiblerObjective(
                wmodel,
                kappas_t,
                inference,
                N_updates,
                lr=self._args["lr_m_step"],
                lr_vi=1e-3,
                device=device,
                dtype=dtype,
            )

            objective.representation = GaussianMultivariateTargetDistribution(
                mean=mean, cov=cov
            )

        return objective

    def discriminative(self, trained=False):

        surrogate = DiscriminativeCNN(
            nx=64,
            filters=[4, 8, 12, 16],
            dim_out=len(self.target),
            dropout=0.05,
            num_additional_full_layers=1,
            make_deterministic=False,
            gate=False,
        )

        if trained:
            surrogate.load(self._folder + "/surrogate/trained.pt")

        surrogate.eval()

        return surrogate

    @classmethod
    def default_settings(cls, DEBUG=False):

        if DEBUG:
            N_data_acquisitions = 1
            M_steps = 2
            N_em_max = 5
            N_epochs = 20
            N_obj_fct = 64
            N_batches = 10
        else:
            N_data_acquisitions = 6
            M_steps = 1  #
            N_em_max = 300
            N_obj_fct = 64
            N_batches = 20000

        fargs = dict()
        cargs = dict()
        cargs["N_monte_carlo_elbo"] = 32
        cargs["N_em_max_steps"] = N_em_max
        cargs["N_objective_fct_monte_carlo"] = N_obj_fct
        cargs["patience"] = 20  # 10

        cargs["N_training_init"] = 4096
        cargs["N_add"] = 1024
        cargs["N_candidates"] = 4096
        cargs["N_data_acquisitions"] = N_data_acquisitions
        cargs["N_training_baseline"] = [10240]
        cargs["N_validation"] = 256

        if DEBUG:
            cargs["N_add"] = 128
            cargs["N_candidates"] = 256
            cargs["N_validation"] = 128

        cargs["cooldown"] = 125
        cargs["active_strategy"] = "kldlogscore"
        cargs["M_steps"] = M_steps

        cargs["VI_SGD_iterations"] = 45
        cargs["VI_lowrank_dim"] = 50
        cargs["VI_SGD_monte_carlo"] = 1
        cargs["VI_em_samples"] = 1

        targs = dict()
        targs["N_epochs"] = None
        targs["N_batches"] = N_batches
        targs["batch_size"] = 128
        targs["lr"] = 1e-3
        targs["weight_decay"] = 1e-5
        targs["verbose"] = True
        targs["reset_model"] = True

        return fargs, cargs, targs


class ChannelizedFlow64VF03(ChannelizedFlow64):
    def __init__(self, args, path="hdata/ChannelizedFlow64VF03/"):

        if "volumefraction" in args:
            vf = args["volumefraction"]
        else:
            vf = 0.30
            args["volumefraction"] = 0.30

        super().__init__(args, path)

        if not "mobj" in args:
            args["mobj"] = True

        assert 0 <= vf <= 1

    def objective(self, wmodel, cargs: dict, dtype, device):

        assert isinstance(cargs, dict)

        if self._args["mobj"]:
            mean = np.array([7.80, 1.95])
            cov = 1 * np.array([[1.0, 0], [0, 0.005]])
        else:
            mean = np.array([20.5, 3.5])
            cov = np.array([[0.60, -0.03], [-0.03, 0.01]])

        N_samples = 20
        q = Gaussian2D(mean, cov)
        kappas = q.sample(N_samples)
        kappas_t = torch.tensor(kappas, dtype=dtype, device=device)

        N_updates = None
        inference = LowRankVariationalInference(
            cargs["VI_SGD_iterations"],
            cargs["VI_lowrank_dim"],
            cargs["VI_SGD_monte_carlo"],
            batch_size=N_samples,
            N_samples_em=cargs["VI_em_samples"],
        )
        objective = KullbackLeiblerObjective(
            wmodel,
            kappas_t,
            inference,
            N_updates,
            lr=self._args["lr_m_step"],
            lr_vi=1e-3,
            device=device,
            dtype=dtype,
        )
        objective.representation = GaussianMultivariateTargetDistribution(
            mean=mean, cov=cov
        )

        return objective


class Multiphysics32(CaseFactory):
    def __init__(self, args, path="hdata/Multiphysics32/"):

        super().__init__(args, path)

        default_values = {
            "nx": 32,
            "cr": 50,
            "N_training": 2 * 4096,
            "N_validation": 1024,
            "binarize": False,
            "eps": 25,
            "target": ["xx", "E_avg"],
            "N_w": 20,
            "w_max": 35.0,
            "N_g": 8,
            "sigma_w": 0.08 * 35.0,
            "phase_1_thermal": 50.0,
            "phase_0_thermal": 1.0,
            "phase_1_structure": 1.0,
            "phase_0_structure": 50.0,
            "nu_default": 0.3,
            "lr_m_step": 0.25,
        }

        substitute_defaults(default_values, self._args)
        self._N_reference_phi_points = 256

    def phase_high(self):
        raise RuntimeError(
            "Multiphysics-prediction should not call phase_high (htransform -> custom homogenizer settings"
        )

    def phase_low(self):
        raise RuntimeError(
            "Multiphysics-prediction should not call phase_low (htransform -> custom homogenizer settings"
        )

    def phi_target(self, return_index=False):

        phi_target = -5 * np.ones(self.pdim)
        phi_target[-1] = 5

        if return_index:
            ind = np.argmax(phi_target)
            return phi_target, ind

        return phi_target

    def htransform(self):

        return BinarizeDataTransform(
            self._args["nx"],
            cutoff=self.cutoff,
            phase_low=0.0,
            phase_high=1.0,
            image_channel=False,
        )

    def _hmg(self):

        hmg1 = PoissonHomogenizer.FromImageResolution(self._args["nx"])
        hmg2 = PeriodicHomogenizer.FromImageResolution(self._args["nx"])
        hmg2._nu_default_value = self._args["nu_default"]
        properties = [dict(), dict()]

        properties[0]["high"] = self._args["phase_1_thermal"]
        properties[0]["low"] = self._args["phase_0_thermal"]
        properties[1]["high"] = self._args["phase_1_structure"]
        properties[1]["low"] = self._args["phase_0_structure"]

        hmg = CombinedHomogenizer([hmg1, hmg2], properties=properties)
        return hmg

    def doe(self, type=None):

        doe_training = GaussianDesignOfExperiment(
            dim=self.pdim, N=self._args["N_training"]
        )
        doe_validation = GaussianDesignOfExperiment(
            dim=self.pdim, N=self._args["N_validation"]
        )

        if type is None:
            return doe_training, doe_validation
        elif type.lower() == "training":
            return doe_training
        elif type.lower() == "validation":
            return doe_validation

    def doe_reference(self):

        return GaussianDesignOfExperiment(dim=self.pdim, N=self._N_reference_phi_points)

    def rf(self, force_cpu=True):

        nx = self._args["nx"]

        if force_cpu:
            device = torch.device("cpu")
        else:
            device = self.device

        kernel = GaussianGridKernel(
            2,
            self._args["w_max"],
            self._args["N_w"],
            self._args["N_g"],
            sigma_w=self._args["sigma_w"],
            init_uniform=False,
        )
        kernel = kernel.to(dtype=self.dtype, device=device)
        rf = DifferentiableGaussianRandomField(
            nx, kernel, 2, dtype=self.dtype, device=device
        )
        rf = rf.to(dtype=self.dtype, device=device)
        rfp = SoftmaxRandomFieldParameters(kernel)

        return rf, rfp

    def discriminative(self, trained=False):

        surrogate = DiscriminativeCNN(
            nx=32,
            filters=[4, 8, 16],
            dim_out=len(self.target),
            dropout=0.05,
            num_additional_full_layers=1,
            make_deterministic=False,
            gate=False,
        )

        if trained:
            surrogate.load(self._folder + "/surrogate/trained.pt")

        surrogate.eval()

        return surrogate

    @classmethod
    def default_settings(cls, DEBUG=False):

        if DEBUG:
            N_data_acquisitions = 1
            M_steps = 2
            N_em_max = 5
            N_obj_fct = 64
        else:
            N_data_acquisitions = 8
            M_steps = 5
            N_em_max = 350
            N_obj_fct = 512

        fargs = dict()

        cargs = dict()
        cargs["N_monte_carlo_elbo"] = 256
        cargs["N_em_max_steps"] = N_em_max
        cargs["N_objective_fct_monte_carlo"] = N_obj_fct
        cargs["patience"] = 15

        cargs["N_training_init"] = 768
        cargs["N_add"] = 512
        cargs["N_candidates"] = 4096
        cargs["N_data_acquisitions"] = N_data_acquisitions
        cargs["N_training_baseline"] = None
        cargs["N_validation"] = 256

        cargs["cooldown"] = 100
        cargs["active_strategy"] = "objfctvar"
        cargs["M_steps"] = M_steps

        cargs["VI_SGD_iterations"] = 85
        cargs["VI_lowrank_dim"] = 40
        cargs["VI_SGD_monte_carlo"] = 10
        cargs["VI_em_samples"] = 128 + 64

        targs = dict()
        targs["N_epochs"] = None
        targs["N_batches"] = 20000
        targs["batch_size"] = 128
        targs["lr"] = 1e-3
        targs["weight_decay"] = 1e-5
        targs["verbose"] = True
        targs["reset_model"] = True

        return fargs, cargs, targs

    def objective(self, wmodel, cargs, dtype, device, type=None):

        assert isinstance(cargs, dict)

        inference = LowRankVariationalInference(
            cargs["VI_SGD_iterations"],
            cargs["VI_lowrank_dim"],
            cargs["VI_SGD_monte_carlo"],
            N_samples_em=cargs["VI_em_samples"],
        )

        bounds = np.array([[9.5, 12.5], [7, 10.5]])

        objective = IntervalObjective.FromBounds(
            wmodel,
            inference,
            bounds,
            lr=self._args["lr_m_step"],
            dtype=dtype,
            device=device,
        )
        return objective


class Multiphysics64(Multiphysics32):
    def __init__(self, args):

        if "nx" not in args:
            args["nx"] = 64

        args["N_training"] = 4 * 4096
        args["N_g"] = 10
        args["N_w"] = 35
        args["w_max"] = 65.0
        args["sigma_w"] = 12

        super().__init__(args, "hdata/Multiphysics64/")

    def discriminative(self, trained=False):

        surrogate = DiscriminativeCNN(
            nx=64,
            filters=[4, 8, 12, 16],
            dim_out=len(self.target),
            dropout=0.05,
            num_additional_full_layers=1,
            make_deterministic=False,
            gate=False,
        )

        if trained:
            surrogate.load(self._folder + "/surrogate/trained.pt")

        surrogate.eval()

        return surrogate

    @classmethod
    def default_settings(cls, DEBUG=False):

        if DEBUG:
            N_data_acquisitions = 1
            M_steps = 2
            N_em_max = 5
            N_obj_fct = 64
            N_batches = 20
        else:
            N_data_acquisitions = 4
            M_steps = 5  #
            N_em_max = 350
            N_batches = 15000
            N_obj_fct = 512

        fargs = dict()

        cargs = dict()
        cargs["N_monte_carlo_elbo"] = 256
        cargs["N_em_max_steps"] = N_em_max
        cargs["N_objective_fct_monte_carlo"] = N_obj_fct
        cargs["patience"] = 15  # 10

        cargs["N_training_init"] = 2048
        cargs["N_add"] = 1024
        cargs["N_candidates"] = 4096
        cargs["N_data_acquisitions"] = N_data_acquisitions
        cargs["N_training_baseline"] = None
        cargs["N_validation"] = 256

        cargs["cooldown"] = 100
        cargs["active_strategy"] = "objfctvar"
        cargs["M_steps"] = M_steps

        cargs["VI_SGD_iterations"] = 85
        cargs["VI_lowrank_dim"] = 50
        cargs["VI_SGD_monte_carlo"] = 10
        cargs["VI_em_samples"] = 128 + 64

        targs = dict()
        targs["N_epochs"] = None
        targs["N_batches"] = N_batches
        targs["batch_size"] = 128
        targs["lr"] = 1e-3
        targs["weight_decay"] = 1e-5
        targs["verbose"] = True
        targs["reset_model"] = True

        return fargs, cargs, targs

    def objective(self, wmodel, cargs, dtype, device, type=None):

        assert isinstance(cargs, dict)

        inference = LowRankVariationalInference(
            cargs["VI_SGD_iterations"],
            cargs["VI_lowrank_dim"],
            cargs["VI_SGD_monte_carlo"],
            N_samples_em=cargs["VI_em_samples"],
        )

        bounds = np.array([[8.5, 11], [6.75, 9]])

        objective = IntervalObjective.FromBounds(
            wmodel,
            inference,
            bounds,
            lr=self._args["lr_m_step"],
            dtype=dtype,
            device=device,
        )
        return objective


class DataFactory(object):
    def __init__(self, case_idf, case_args, num_samples_per_phi, DOE):

        self._DOE = DOE
        self._case_idf = case_idf
        self._case_args = case_args
        self._num_samples_per_phi = num_samples_per_phi

    def __call__(
        self,
        retain_X_g=False,
        retain_X_b=False,
        retain_phi=False,
        mpi_num_workers=None,
        shared_memory_parallelization=False,
        calculate_features=False,
        feature_dict=None,
    ):

        if calculate_features or feature_dict:
            raise NotImplementedError(
                "Pre-computation of features has not yet been implemented"
            )

        results = dict()

        results["kappa"] = list()

        if retain_X_g:
            results["X_g"] = list()
        if retain_X_b:
            results["X_b"] = list()
        if retain_phi:
            results["phi"] = list()

        if mpi_num_workers:
            assert isinstance(mpi_num_workers, int) and mpi_num_workers >= 1
            num_workers_used = min(mpi_num_workers, len(self._DOE))
            pool = MPIPoolExecutor(max_workers=num_workers_used)
        elif shared_memory_parallelization:
            # shared memory parallel
            pool = ProcessPoolExecutor(8)
        else:
            pool = DummyProcessPool()

        futures = list()
        for n, phi in enumerate(self._DOE):
            futures.append(
                pool.submit(
                    exec,
                    self._case_idf,
                    self._case_args,
                    phi,
                    self._num_samples_per_phi,
                    retain_X_b,
                    retain_X_g,
                    retain_phi,
                )
            )

        tracker = PoolTracker(futures)
        T_SLEEP_INTERVAL = 10 if not mpi_num_workers else 30
        tracker.wait(verbose=True, T_SLEEP_INTERVAL=T_SLEEP_INTERVAL)
        pool.shutdown(wait=True)
        is_backend = backend()
        for n, future in tqdm(enumerate(futures), disable=is_backend):

            if is_backend:
                print("Future {} / {}".format(n, len(futures)))

            try:
                res_ = future.result()
                if not (isinstance(res_, str) and res_ == "PURGE"):
                    for key in res_.keys():
                        results[key].append(res_[key])

            except Exception as err:

                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Encoundered error in future {} / {}".format(n, len(futures)))
                print(err)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        return results

    def training_data(self, mpi_num_workers=None, shared_memory_parallelization=False):

        RES = self(
            retain_X_g=True,
            retain_X_b=True,
            retain_phi=True,
            mpi_num_workers=mpi_num_workers,
            shared_memory_parallelization=shared_memory_parallelization,
        )

        assert (
            self._num_samples_per_phi == 1
        ), "the following code assumes only one data point per phi has been generated"
        kappa_keys = RES["kappa"][0][0].keys()
        kappa = dict()
        for kappa_key in kappa_keys:
            kappa[kappa_key] = torch.zeros(len(self._DOE))

        X_g = torch.zeros((len(self._DOE),) + RES["X_g"][0][0].shape)
        X_b = torch.zeros((len(self._DOE),) + RES["X_b"][0][0].shape)
        phi = torch.zeros(len(self._DOE), len(RES["phi"][0][0]))

        for n in range(len(self._DOE)):

            for j, mkey in enumerate(kappa_keys):
                kappa[mkey][n] = RES["kappa"][n][0][mkey]

            X_g[n, :, :] = RES["X_g"][n][0]
            X_b[n, :, :] = RES["X_b"][n][0]
            phi[n, :] = torch.tensor(RES["phi"][n][0])

        return kappa, X_g, X_b, phi


def exec(
    case_idf, case_args, phi, num_samples_per_phi, retain_X_b, retain_X_g, retain_phi
):

    factory = CaseFactory.FromIdentifier(case_idf, case_args)

    rf, _ = factory.rf()
    hmg = factory.hmg()
    htransform = factory.htransform()

    rf.set_phi(phi)

    results = dict()
    results["kappa"] = list()

    if retain_X_g:
        results["X_g"] = list()
    if retain_X_b:
        results["X_b"] = list()
    if retain_phi:
        results["phi"] = list()

    with torch.no_grad():

        for n in range(num_samples_per_phi):

            X_g = rf.rsample()
            X_b = htransform(X_g)
            assert X_b.shape == X_g.shape

            kappa = hmg.homogenize_img(X_b, AcknowledgeRaw=True)

            results["kappa"].append(kappa)

            if retain_X_g:
                results["X_g"].append(X_g)
            if retain_X_b:
                results["X_b"].append(X_b)
            if retain_phi:
                results["phi"].append(phi)

    return results
