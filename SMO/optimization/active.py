import torch
import numpy as np
from lamp.data import CustomTensorDataset
from torch.utils.data import DataLoader
from typing import List


def setup_active_learner(
    factory,
    data_tr,
    data_val,
    wmodel,
    objective,
    N_outer,
    N_add,
    N_candidates,
    learning_strategy="objvar",
):

    # utility function
    dtype = objective.dtype
    device = objective.device

    data_dict = dict()
    data_dict["X_cnn"] = data_tr["X_cnn"].to(dtype=dtype, device=device)
    data_dict["X_g"] = (
        data_tr["X_g"].to(dtype=dtype, device=device).to(dtype=dtype, device=device)
    )
    data_dict["kappa"] = data_tr["kappa"].to(dtype=dtype, device=device)

    dhandler = DataHandler(data_dict, wmodel, dtype=dtype, device=device)
    active = ActiveLearner(
        wmodel,
        objective,
        factory,
        dhandler,
        iterations=N_outer,
        strategy=learning_strategy,
    )
    wmodel.datahandler = dhandler

    active.N_add = N_add
    active.N_candidates = N_candidates
    wmodel.datahandler = dhandler

    return active, dhandler


def logscore_acquisition_fct_kld(
    kappas_s: torch.Tensor, mean: torch.Tensor, logsigma: torch.Tensor
):

    assert (
        isinstance(kappas_s, torch.Tensor)
        and isinstance(mean, torch.Tensor)
        and isinstance(logsigma, torch.Tensor)
    )
    assert kappas_s.ndim == mean.ndim == logsigma.ndim
    assert kappas_s.shape[1] == mean.shape[1] == logsigma.shape[1]

    q = torch.distributions.Normal(mean, torch.exp(logsigma))
    qi = torch.distributions.Independent(q, 1)

    return qi.log_prob(kappas_s.unsqueeze(1)).mean(0)


class ActiveLearner(object):
    def __init__(
        self,
        model,
        objective,
        factory,
        datahandler,
        iterations=None,
        N_add=None,
        N_candidates=None,
        strategy="objfctvar",
    ):

        assert strategy in ["objfctvar", "var", "random", "kldlogscore"]
        self._strategy = strategy

        assert isinstance(datahandler, DataHandler)
        assert isinstance(iterations, int)

        self._model = model
        self._objective = objective
        self._datahandler = datahandler
        self._factory = factory

        self.N_add = N_add
        self.N_candidates = N_candidates

        self._hmg = factory.hmg(constrain_to_target=True)
        self._htransform = factory.htransform()
        self._dtransform = factory.dtransform()

        self._criteria = list()
        self._dfactor = list()

        self._iterations = iterations
        self._iteration_counter = 0

    @property
    def N_data(self):
        return len(self._datahandler)

    @property
    def dfactor(self):
        return self._dfactor

    def __bool__(self):
        return self._iteration_counter < self._iterations

    @torch.no_grad()
    def _employ_strategy(self, Pt, N_add, N_candidates):

        if self._strategy == "objfctvar":

            assert hasattr(
                self._objective, "var"
            ), "the supplied objective instance is not able to calculate variances via var()"
            vars = self._objective.var(Pt)
            sorted, isort = torch.sort(vars, descending=True)
            index = isort[0:N_add].tolist()

            total_var = torch.sum(vars).item()
            partial_var = torch.sum(sorted[0:N_add]).item()
            disproportionality_factor = (partial_var / total_var) / (
                N_add / N_candidates
            )

        elif self._strategy == "var":

            raise DeprecationWarning

        elif self._strategy == "random":

            raise DeprecationWarning

        elif self._strategy == "kldlogscore":

            assert hasattr(
                self._objective, "kappas_target"
            ), "the supplied objective instance does not specify samples from a target distribution (KLD)"

            mean, logsigma = self._model.pipeline(Pt, mode="transformed_phaseangles")
            loglkl = logscore_acquisition_fct_kld(
                self._objective.kappas_target, mean, logsigma
            )
            assert loglkl.ndim == 1 and len(loglkl) == Pt.shape[0]

            sorted, isort = torch.sort(loglkl, descending=True)
            index = isort[0:N_add].tolist()
            disproportionality_factor = torch.sqrt(
                loglkl.var() / (loglkl.mean() ** 2 + 1e-6)
            ).item()

        else:
            raise NotImplementedError

        return index, disproportionality_factor

    @torch.no_grad()
    def __call__(
        self, N_add=None, N_candidates=None, *, enable_random_monkey=False, Pt=None
    ):

        N_candidates = self.N_candidates if N_candidates is None else N_candidates
        N_add = self.N_add if N_add is None else N_add
        assert N_candidates is not None and N_add is not None
        assert N_add < N_candidates, "we require N_add < N_candidates"

        if Pt is None:
            Pt = self._model.rf.sample_transformed_phase_angles(N_candidates)
        else:
            assert N_candidates == Pt.shape[0]

        index, disproportionality_factor = self._employ_strategy(
            Pt, N_add, N_candidates
        )

        self._dfactor.append(disproportionality_factor)

        X_g, X_cnn, kappa = self._homogenize(Pt[index])

        self._datahandler.add_data(X_cnn=X_cnn, X_g=X_g, kappa=kappa)

        self._iteration_counter += 1

    @torch.no_grad()
    def _homogenize(self, Pt: torch.Tensor):

        assert Pt.ndim == 2

        Xg = self._model.rf.rsample_transform(Pt)
        Xh = self._htransform(Xg)
        Xcnn = self._dtransform(Xg)

        kappas = self._hmg.homogenize_img(Xh, AcknowledgeRaw=True)
        assert isinstance(kappas, list) and len(kappas) > 1

        kappas = torch.tensor(
            [[kappa[t] for t in self._factory.target] for kappa in kappas],
            dtype=Pt.dtype,
            device=Pt.device,
        )

        return Xg, Xcnn, kappas


class DataHandler(object):
    def __init__(self, data, model, *, dtype=None, device=None):

        assert isinstance(data, dict)
        assert isinstance(dtype, torch.dtype)
        assert isinstance(device, torch.device)

        self._data = data
        self._wmodel = model
        self._dtype = dtype
        self._device = device

        self._N = self._check(data, enforce_device=True, enforce_dtype=True)
        # currently not used
        self._metadata = torch.zeros(self._N, dtype=torch.long, device=device)

        self._info = [self._N]
        self._phi = [None]

        self._dataloader = None

    @property
    def N_acquisitions(self):

        return len(self._info) - 1

    def export(self, device="cpu", entries=None, clone=True) -> list:

        assert device in ["cpu", "cuda:0"]

        if entries is None:
            entries = ["X_g", "kappa"]

        for entry in entries:
            assert (
                entry in self._data.keys()
            ), "cannot export {} because do not have data".format(entries)

        N = self.N_acquisitions + 1

        datalist = [dict() for n in range(N)]

        for n, dataitem in enumerate(datalist):
            for key in entries:
                data = self.inspect_samples(n, type=key).to(device=torch.device(device))
                if clone:
                    data = data.clone().detach()
                dataitem[key] = data

        return datalist

    def inspect_samples(self, step, type="X_cnn"):

        if step == -1:
            step = len(self._info) - 1

        assert step < len(self._info)

        info_ = np.array(self._info)
        lower = np.sum(info_[0:step])
        upper = np.sum(info_[0 : step + 1])
        S = self._data[type][lower:upper]
        assert S.shape[0] == self._info[step]

        return S

    @property
    def model(self):
        return self._wmodel

    def datadict(self, data_val: dict):

        data = dict()
        data["X"] = self._data["X_cnn"]
        data["Y"] = self._data["kappa"]
        data["X_val"] = data_val["X_cnn"].to(dtype=self._dtype, device=self._device)
        data["Y_val"] = data_val["kappa"].to(dtype=self._dtype, device=self._device)

        return data

    def dataloader(
        self, keys: List[str], bs: int, shuffle=True, enforce_divisible=False
    ):

        assert isinstance(keys, list)
        assert all(isinstance(key, str) for key in keys)
        assert isinstance(bs, int) and bs > 0
        assert (
            not enforce_divisible or self._N % bs == 0
        ), "the batch size {} does not evenly partition {} data points".format(
            bs, self._N
        )

        dataset = CustomTensorDataset(*(self._data[key] for key in keys))
        return DataLoader(dataset=dataset, batch_size=bs, shuffle=shuffle)

    def __len__(self):
        return self._N

    def _check(self, data, enforce_device=True, enforce_dtype=True):

        # check that the data to be added is valid
        N = None
        for item in data.values():
            assert isinstance(
                item, torch.Tensor
            ), "trying to add data entry which is of type {} (requires torch.Tensor)".format(
                type(item)
            )
            if N is None:
                N = item.shape[0]
            else:
                assert item.shape[0] == N
            assert (
                item.dtype == self._dtype or not enforce_dtype
            ), "The data supplied to DataHandler does not match set dtype"
            assert (
                item.device == self._device or not enforce_device
            ), "The data supplied to DataHandler does not match set device"

        return N

    def _update(self):

        pass

    def add_data(self, **kwargs):

        N_add = self._check(kwargs)

        assert set(kwargs.keys()) == set(self._data.keys())

        for kwarg in kwargs.keys():
            self._data[kwarg] = torch.cat(
                (
                    self._data[kwarg],
                    kwargs[kwarg].to(dtype=self._dtype, device=self._device),
                ),
                0,
            )

        self._N += N_add

        self._info.append(N_add)
        self._phi.append(self._wmodel.rf.get_phi())

        self._update()
