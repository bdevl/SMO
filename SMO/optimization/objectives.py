import warnings
from copy import deepcopy
from math import ceil
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import bisect
from scipy.stats import kde
from scipy.stats import multivariate_normal
from scipy.stats import norm as singlevariate_normal
from torch.distributions import Normal
import matplotlib.patches as patches

import SMO.factories
import lamp.modules
from SMO.discriminative import DiscriminativeTrainer, DiscriminativeCNN
from SMO.optimization.active import DataHandler
from lamp.utils import DiagonalGaussianLogLikelihood
from lamp.utils import reparametrize


class WrapperModel(lamp.modules.BaseModule):
    def __init__(
        self,
        surrogate,
        rf,
        rfp,
        dtransform,
        kappa_target=None,
        target=None,
        hmg=None,
        htransform=None,
    ):

        super().__init__()

        self._discriminative = surrogate
        self._rf = rf
        self._rfp = rfp
        self._dtransform = dtransform
        self._kappa_target = kappa_target
        self._target = target
        self._datahandler = None
        self._autotrain_args = None
        self._r2_achieved = None
        self._hmg = hmg
        self._htransform = htransform

    @classmethod
    def FromFactory(cls, factory, surrogate=None, *, dtype=None, device=None):

        # conveniency function
        assert dtype is not None and device is not None

        if surrogate is None:
            surrogate = factory.discriminative()

        surrogate = surrogate.to(dtype=dtype, device=device)

        rf, rfp = factory.rf()
        rf = rf.to(dtype=dtype, device=device)

        dtransform = factory.dtransform()

        target = factory.target

        hmg = factory.hmg()

        htransform = factory.htransform()

        return cls(
            surrogate,
            rf,
            rfp,
            dtransform,
            target=target,
            hmg=hmg,
            htransform=htransform,
        )

    @property
    def hmg(self):
        return self._hmg

    def state(self):
        return self._discriminative.state_dict()

    @property
    def datahandler(self):
        return self._datahandler

    @datahandler.setter
    def datahandler(self, val):

        assert isinstance(val, DataHandler)
        assert (
            val.model == self
        ), "The model set for the datahandler is not equal to self."
        self._datahandler = val

    def autotrain_failsafe(self, *args, threshold=0.0, attempts=2, **kwargs):

        assert kwargs["reset_model"], "assumes that surrogate is reset every time"

        for n in range(attempts):

            self.autotrain(*args, **kwargs)
            success = not np.any(np.array(self._r2_achieved) < threshold)

            if success and n < attempts:
                break

            if not success and n == attempts - 1:
                raise RuntimeError("Training failure")

    def autotrain(self, args: dict = None):

        if args is not None:
            self._autotrain_args = args
        else:
            return self.train_model(**self._autotrain_args)

    def train_model(
        self,
        N_epochs,
        init_dict,
        data_val,
        batch_size,
        lr,
        weight_decay=1e-6,
        verbose=True,
        reset_model=True,
        N_batches=None,
    ):

        assert (
            self._datahandler is not None
        ), "in order to execute training, the model wrapper requires a datahandler"

        if N_epochs is None:
            assert N_epochs is None, "cannot specify both N_epochs and N_batches"
            num_batches_per_epoch = int(len(self._datahandler) / batch_size)
            N_epochs = ceil(N_batches / num_batches_per_epoch)

        training_state = self.training
        self.train()
        assert self._discriminative.training

        if reset_model:
            self._discriminative.load_state_dict(deepcopy(init_dict))
        trainer = DiscriminativeTrainer(
            self._discriminative,
            self._datahandler.datadict(data_val),
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            make_deterministic=False,
            shuffle=True,
        )
        trainer.train(N_epochs, verbose=verbose, output_interval=50)

        self._r2_achieved = trainer.r2_separate()

        if not training_state:
            self.eval()
            assert not self._discriminative.training

        return trainer

    @property
    def rf(self):
        return self._rf

    @property
    def rfp(self):
        return self._rfp

    @property
    def target(self):
        if self._target is None:
            raise RuntimeError("The target property has not been set for the model")
        return self._target

    @property
    def kappa_target(self):
        return self._kappa_target

    @kappa_target.setter
    def kappa_target(self, val):
        assert isinstance(val, torch.Tensor)
        self._kappa_target = val

    @property
    def differentiable(self):
        return self._dtransform.differentiable

    def ensure_differentiable(self):

        assert self.differentiable, "The model is not differentiable."

    def forward(self, X_cnn: torch.Tensor):

        return self._discriminative(X_cnn)

    def pipeline(self, X: torch.Tensor, *, mode: str = None):

        mode = mode.lower()
        assert mode is not None

        if mode == "transformed_phaseangles":
            assert (
                X.shape[1] == self._rf.kernel.phase_angle_dim
            ), "The phase angle tensor does not match (in the dimension of the phase angles)"
            return self._pipeline(X)
        else:
            raise ValueError(
                "The mode {} is not available to evaluate the model".format(mode)
            )

    def _pipeline(self, P):

        X_g = self.rf.rsample_transform(P)

        if X_g.dim() < 4:
            X_g = X_g.unsqueeze(1)

        X = self._dtransform(X_g)

        mu, logsigma = self._discriminative(X)
        return mu, logsigma

    def pipeline_Xg(self, X_g):

        if X_g.dim() < 4:
            X_g = X_g.unsqueeze(1)

        X = self._dtransform(X_g)

        mu, logsigma = self._discriminative(X)
        return mu, logsigma

    def LogLikelihood(self, P, kappa=None, reduce=torch.sum):

        kappa = self._kappa_target if kappa is None else kappa

        mu, logsigma = self._pipeline(P)

        return DiagonalGaussianLogLikelihood(kappa, mu, 2 * logsigma, reduce=reduce)

    @torch.no_grad()
    def sample_reference_distribution(
        self, N: int = 64, *, phi=None, q=None, return_X_cnn=False
    ):

        assert isinstance(N, int) and N > 1
        assert phi is None or isinstance(phi, (list, np.ndarray))

        if q is not None:
            raise NotImplementedError

        assert (
            self._hmg is not None
            and self._htransform is not None
            and self._target is not None
        ), "either _hmg /_htransform / _target have not been set for WrapperModel"

        if phi is not None:
            phi_orig = deepcopy(self._rf.get_phi())
            self._rf.set_phi(phi)

        Xg = self.rf.rsample_batch(N)
        Xh = self._htransform(Xg)

        # list of dicts
        kappas = self._hmg.homogenize_img(Xh, AcknowledgeRaw=True)
        assert isinstance(kappas, list) and len(kappas) > 1

        # convert to torch.tensor
        kappas = torch.tensor(
            [[kappa[t] for t in self.target] for kappa in kappas],
            dtype=Xg.dtype,
            device=Xg.device,
        )

        if phi is not None:
            self._rf.set_phi(phi_orig)

        if return_X_cnn:
            Xcnn = self._dtransform(Xg)
            return Xcnn, kappas
        else:
            return kappas

    @torch.no_grad()
    def sample_predicted_distribution(
        self, N: int = 1024, *, phi=None, q=None
    ) -> torch.Tensor:

        assert isinstance(N, int) and N > 1
        assert phi is None or isinstance(phi, (list, np.ndarray))

        if phi is not None:
            phi_orig = deepcopy(self._rf.get_phi())
            self._rf.set_phi(phi)

        if q is not None:
            Pt = q.rsample(N)
            assert Pt.ndim == 2 or Pt.shape[1] == 1, "not implemented"

            if Pt.shape[1] == 1:
                Pt = Pt.squeeze(1)

            assert Pt.ndim == 2
            X_g = self._rf.rsample_transform(Pt)
        else:
            X_g = self._rf.rsample_batch(N)
        kappa_mean, kappa_logsigma = self.pipeline_Xg(X_g)
        kappa_samples = reparametrize(kappa_mean, kappa_logsigma)
        assert kappa_samples.ndim == 2 and kappa_samples.shape[0] == N

        if phi is not None:
            self._rf.set_phi(phi_orig)

        return kappa_samples

    @torch.no_grad()
    def plot_joint_distribution_2d(
        self,
        N,
        int=1024,
        *,
        nbins=40,
        offset=1,
        kmin=None,
        kmax=None,
        q=None,
        phi=None,
        **kwargs
    ):

        kappa_samples = (
            self.sample_predicted_distribution(N, q=q, phi=phi).cpu().detach().numpy()
        )

        # establish bounds
        kappas = kappa_samples
        assert (
            isinstance(kappas, np.ndarray) and kappas.ndim == 2 and kappas.shape[1] == 2
        ), "density estimate failed"
        kmin = (
            np.stack(kappas).reshape(-1, kappas.shape[1]).min(0) - offset
            if kmin is None
            else kmin
        )
        kmax = (
            np.stack(kappas).reshape(-1, kappas.shape[1]).max(0) + offset
            if kmax is None
            else kmax
        )

        # density estimate
        kde_ = kde.gaussian_kde([kappas[:, 0], kappas[:, 1]])
        xi, yi = np.mgrid[
            kmin[0] : kmax[0] : nbins * 1j, kmin[1] : kmax[1] : nbins * 1j
        ]
        zi = kde_(np.vstack([xi.flatten(), yi.flatten()]))
        h1 = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap="magma")
        plt.colorbar(h1)

    def plot_marginal_distribution(
        self, N: int = 1024, *, q=None, phi=None, use_reference=False
    ):

        if use_reference:
            kappa_samples = (
                self.sample_reference_distribution(N, q=q, phi=phi)
                .cpu()
                .detach()
                .numpy()
            )
        else:
            kappa_samples = (
                self.sample_predicted_distribution(N, q=q, phi=phi)
                .cpu()
                .detach()
                .numpy()
            )

        # quick and dirty way to look at results
        dim_kappa = kappa_samples.shape[1]
        fig, axi = plt.subplots(1, dim_kappa, figsize=(4 + dim_kappa, 4))

        for dk in range(dim_kappa):
            if dim_kappa > 1:
                plt.sca(axi[dk])
            plt.hist(kappa_samples[:, dk], density=True, alpha=0.65)
            plt.grid(True)
            plt.title(r"$\kappa_{}$".format(dk + 1))

        return fig, axi


class IntervalHelper(object):
    def __init__(self):
        pass


class AbstractObjective(object):
    def __init__(self, model, dtype, device, lr=0.25):

        assert isinstance(model, WrapperModel)
        assert isinstance(
            dtype, torch.dtype
        ), "Need to provide valid dtype to objective"
        assert isinstance(
            device, torch.device
        ), "Need to provide valid device to objective"

        model.ensure_differentiable()
        self._model = model

        self._optim = torch.optim.Adam(self._params, lr=lr)

        self._cache = dict()
        self._cache["initialized"] = False
        self._cache["counter"] = 0

        self._dtype = dtype
        self._device = device

        self._lr = lr

    @property
    def tempering_completed(self):
        raise NotImplementedError(
            'method "tempering_completed" needs to be implemented by child class'
        )

    @property
    def wmodel(self):
        # wrapper model
        return self._model

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, val):
        assert val > 0 and isinstance(val, float)
        for param_group in self._optim.param_groups:
            param_group["lr"] = val
        self._lr = val

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def rf(self):
        return self._model.rf

    @property
    def rfp(self):
        return self._model.rfp

    @property
    def _params(self):
        return self.rf.kernel.parameters()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def assess(self, kappa: torch.Tensor = None, N=None, source="rf") -> float:

        if kappa is None:
            raise NotImplementedError("need to call homogenizer")

        return self._assess(kappa)

    @torch.no_grad()
    def assess_model_belief(self, N, source="rf"):

        assert (
            isinstance(N, int) and N > 1
        ), "Need to specify a valid number of microstructure samples to assess"

        raise NotImplementedError

    def _assess(self, kappa: torch.Tensor) -> float:

        raise NotImplementedError

    def estimate_reference(self, hmg_fct: Callable, N_monte_carlo: int) -> float:

        raise NotImplementedError

    def _LogPrior(
        self, Pt: torch.Tensor, return_separate_batch_values=False
    ) -> torch.Tensor:

        if return_separate_batch_values:
            return -0.5 * torch.sum(Pt ** 2, 1)
        else:
            return -0.5 * torch.sum(Pt ** 2)

    def init(self, enforce=False, *args, **kwargs):

        if not self._cache["initialized"] or enforce:
            self._init(*args, **kwargs)
            self._cache["initialized"] = True

    def _init(self, *args, **kwargs):
        raise NotImplementedError("Absctract method not yet implemented.")

    def step(self, *args, **kwargs):

        self._cache["counter"] += 1
        self._step(*args, **kwargs)

    def _step(self, *args, **kwargs):
        raise NotImplementedError("Absctract method not yet implemented.")

    def E(self, *args, **kwargs):

        assert (
            not self._model.training
        ), "Trying to execute the E-step with model in training() mode"
        return self._E(*args, **kwargs)

    def _E(self, *args, **kwargs):
        raise NotImplementedError("Absctract method not yet implemented.")

    def M(self, *args, **kwargs):

        M_steps = kwargs.pop("M_steps", 1)

        for n in range(M_steps):
            assert (
                not self._model.training
            ), "Trying to execute the M-step with model in training() mode"
            self._optim.zero_grad(set_to_none=True)
            if n > 0:

                self._E_sample()

            self._M(*args, **kwargs)

            self._optim.step()
            self.rfp.clamp()

        if kwargs.pop("return_grad", False):
            raise NotImplementedError

    def _M(self, *args, **kwargs):
        raise NotImplementedError("Abstract method not yet implemented")

    def cleanup(self):

        self._optim.zero_grad(set_to_none=True)


class LessThanDomainObjective(AbstractObjective):
    def __init__(
        self,
        model,
        inference,
        less_than_functional,
        N_init_samples=512,
        target_fraction=0.75,
        *,
        dtype=None,
        device=None
    ):

        super().__init__(model, dtype=dtype, device=device)

        self._inference = inference
        self._ltf = less_than_functional
        self._thrs = None

        self._dim_kappa = len(self._model.target)

        self._E_samples = None
        dim = model.rf.kernel.phase_angle_dim + self._dim_kappa

        def logpot(Z):

            kappa = Z[:, 0 : self._dim_kappa]
            Pt = Z[:, self._dim_kappa :]

            return self._J(Pt, kappa)

        self._inference.init(logpot=logpot, dim=dim)
        self._inference.to(dtype=self._dtype, device=self._device)
        self._N_init_samples = N_init_samples
        self._target_fraction = 0.25

        self._compliance = list()
        self._elbo_comp = list()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def _init(self, *args, **kwargs):

        X_g = self._model.rf.rsample_batch(self._N_init_samples)
        kappa_mean, kappa_logsigma = self._model.pipeline_Xg(X_g)
        kappa_samples = reparametrize(kappa_mean, kappa_logsigma)

        c = self._ltf(kappa_samples)
        c_t = torch.quantile(c, self._target_fraction)

        if c_t <= 0:
            print(
                "Init thrs to zero. No tempering will be done | c_t = {:.2f}.".format(
                    c_t
                )
            )
            c_t = 0

        self._thrs = c_t

    def _step(self, *args, **kwargs):

        self._temper(*args, **kwargs)

    @torch.no_grad()
    def _temper(self, verbose=False):

        if self._E_samples is None:
            raise RuntimeError("Trying to do a ESS tempering step without prior E-step")

        theta_hat_samples = self._E_samples

        assert (
            0 < self._target_fraction < 1
        ), "The desired target fraction needs to be in [0,1]"

        if self._thrs == 0:
            return

        thrs_last = self._thrs

        kappa_mu, kappa_logsigma = self._model.pipeline(
            theta_hat_samples, mode="transformed_phaseangles"
        )
        kappa_samples = reparametrize(kappa_mu, kappa_logsigma)
        c = self._ltf(kappa_samples)

        thrs = torch.quantile(kappa_samples, self._target_fraction)

        if thrs <= 0:
            print("Tempering has concluded. Setting thrs equal to 0.")
            self._thrs = 0
            return

        if thrs > thrs_last:
            warnings.warn(
                "Tempering tried to shift domain backwards (c_t : {:.2f} -> {:.2f})".format(
                    thrs_last, thrs
                )
            )
            return

        self._thrs = thrs

    def _J(self, Pt: torch.Tensor, kappa: torch.Tensor, track=False):

        kappa_mu, kappa_logsigma = self._model.pipeline(
            Pt, mode="transformed_phaseangles"
        )

        logp = DiagonalGaussianLogLikelihood(
            kappa, kappa_mu, 2 * kappa_logsigma, reduce=torch.sum
        )
        assert logp.ndim == 0
        log_prior = self._LogPrior(Pt)
        assert log_prior.ndim == 0
        eps = 0.000001
        sfc = self._ltf(kappa) < self._thrs
        ltf_ = torch.log((sfc).float().clamp(eps))

        if track:
            self._compliance.append(sfc.float().mean())
            self._elbo_comp.append(ltf_.sum().item())
            print("=> Compliance: {:.2f}".format(self._compliance[-1]))

        if track:
            print("logp: {:.2f}".format(logp.item()))
            print("log_prior: {:.2f}".format(log_prior.item()))
            print("ltf: {:.2f}".format(ltf_.sum().item()))

        return logp + log_prior + ltf_.sum()

    def _E(self):
        self._E_samples = self._inference.execute()

    def _M(self) -> float:

        Z = self._E_samples.detach()
        kappa = Z[:, 0 : self._dim_kappa]  # functional
        Pt = Z[:, self._dim_kappa :]  # phase angles
        bs = Z.shape[0]
        assert Pt.ndim == 2
        J = -self._J(Pt, kappa, track=True) / bs
        J.backward()
        return -J.item()


class IntervalRepresentation(object):
    def __init__(self, bounds: np.ndarray):

        self._bounds = bounds

    def mark_1d(self):
        raise NotImplementedError

    def mark_2d(self, color="g", linewidth=1.5, ax=None, **kwargs):

        assert isinstance(self._bounds, np.ndarray) and self._bounds.shape == (
            2,
            2,
        ), "assumes 2D"
        midpoint = tuple(self._bounds.mean(1))
        extensions = self._bounds[:, 1] - self._bounds[:, 0]
        xy = np.array(midpoint) - 0.5 * extensions
        xy = tuple(xy)  # lower left corner, anchor point
        rect = patches.Rectangle(
            xy,
            extensions[0],
            extensions[1],
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            **kwargs
        )
        ax = ax if ax is not None else plt.gca()
        ax.add_patch(rect)


class IntervalObjective(AbstractObjective):
    def __init__(
        self,
        model,
        inference,
        interval,
        tempering="ess",
        N_init_samples=512,
        ess_reduction=0.80,
        init_symmetric=False,
        lr=0.25,
        *,
        dtype=None,
        device=None
    ):

        super().__init__(model, lr=lr, dtype=dtype, device=device)
        assert isinstance(interval, TemperedIntervalManager)

        self._interval = interval
        assert tempering in [
            "ess"
        ], "Currently only using the ESS for tempering is supported."
        self._tempering = tempering

        self._inference = None
        self.set_inference(inference)
        self._E_samples = None

        self._N_init_samples = N_init_samples
        self._init_symmetric = init_symmetric
        self._ess_reduction = ess_reduction

        self.representation = IntervalRepresentation(interval.bounds.copy())

    @property
    def tempering_completed(self):

        return self._interval.alpha == 1

    def set_inference(self, inference):

        self._inference = inference
        self._inference.init(
            logpot=lambda Pt: self._LogPrior(Pt) + self.LogLikelihood(Pt),
            dim=self._model.rf.kernel.phase_angle_dim,
        )
        self._inference.to(dtype=self._dtype, device=self._device)

    @classmethod
    def FromBounds(cls, model, inference, bounds: np.ndarray, *args, **kwargs):

        interval = TemperedIntervalManager(bounds)
        return cls(model, inference, interval, *args, **kwargs)

    @torch.no_grad()
    def _init(self):

        X_g = self._model.rf.rsample_batch(self._N_init_samples)
        kappa_mean, kappa_logsigma = self._model.pipeline_Xg(X_g)
        kappa_samples = reparametrize(kappa_mean, kappa_logsigma)

        if self._init_symmetric:
            self._interval.init_symmetric(kappa_samples, quantile=0.05)
        else:
            self._interval.init(kappa_samples, quantile=0.05)

    def __call__(self, Pt: torch.Tensor, return_individual_probabilities=False):

        # X is size(batch_size, py, px) and involves e.g. the soft-threshholded dtransform(X_g)
        kappa_mean, kappa_logsigma = self._model.pipeline(
            Pt, mode="transformed_phaseangles"
        )

        q = torch.distributions.Normal(kappa_mean, torch.exp(kappa_logsigma))

        lower = self._interval.lower_bounds(dtype=Pt.dtype, device=Pt.device)
        upper = self._interval.upper_bounds(dtype=Pt.dtype, device=Pt.device)

        diff = q.cdf(upper) - q.cdf(lower)
        diff = torch.clamp(diff, min=1e-5)

        if return_individual_probabilities:
            return diff
        else:
            return torch.prod(diff, 1)

    # returns scalar torch.tensor
    def LogLikelihood(self, Pt, return_separate_batch_values=False) -> torch.Tensor:

        diff = self(Pt, return_individual_probabilities=True)

        if return_separate_batch_values:
            LogLkl = torch.sum(torch.log(diff), 1)
        else:
            LogLkl = torch.sum(torch.log(diff))

        return LogLkl

    def _step(self, *args, **kwargs):

        self._temper(*args, **kwargs)

    def _assess(self, kappa: torch.Tensor, active_interval=False):

        J = self._interval.fraction_within(
            kappa, active_interval, return_individual=False
        )
        assert isinstance(J, float) and 0 <= J <= 1

        return J

    @torch.no_grad()
    def var(self, Pt: torch.Tensor, N_monte_carlo=None) -> torch.Tensor:

        if N_monte_carlo is not None:
            pass

        probs = self(Pt, return_individual_probabilities=True)

        q = torch.prod(probs, 1)
        vars = q * (1 - q)

        return vars

    def _temper(self, verbose=False):

        if self._E_samples is None:
            raise RuntimeError("Trying to do a ESS tempering step without prior E-step")

        theta_hat_samples = self._E_samples
        ess_reduction = self._ess_reduction

        assert 0 < ess_reduction < 1

        if self._interval.alpha == 1:
            return

        alpha_last = self._interval.alpha

        with torch.no_grad():

            kappa_mu, kappa_logsigma = self._model.pipeline(
                theta_hat_samples, mode="transformed_phaseangles"
            )
            log_prior = self._LogPrior(
                theta_hat_samples, return_separate_batch_values=True
            )
            log_posterior = self.LogLikelihood(
                theta_hat_samples, return_separate_batch_values=True
            )
            logp = log_prior + log_posterior
            q = Normal(kappa_mu, torch.exp(kappa_logsigma))

            kappa_samples_q = q.sample()

        @torch.no_grad()
        def ess_fct(alpha: float, safeguard=False) -> float:

            lower = self._interval.lower_bounds(
                dtype=kappa_mu.dtype, device=kappa_mu.device, alpha=alpha
            )
            upper = self._interval.upper_bounds(
                dtype=kappa_mu.dtype, device=kappa_mu.device, alpha=alpha
            )
            diff = q.cdf(upper) - q.cdf(lower)
            diff = torch.clamp(diff, min=1e-5)

            limitcase = torch.all(diff < 4.99 * 1e-5).item()
            if safeguard and limitcase:
                return 0
            elif not safeguard and limitcase:
                return 0

            LogLkl = torch.sum(torch.log(diff), 1)
            logp_new = log_prior + LogLkl
            logw = logp_new - logp

            # TODO: evaluate pytorch tensor (on GPU)
            return ESS(logw.detach().cpu().numpy())

        def bisect_fct(alpha):

            if np.abs(alpha_last - alpha) < 1e-8:
                return 1 - ess_reduction
            return ess_fct(alpha) - ess_reduction

        ess_final = ess_fct(1, safeguard=True)

        if ess_final > ess_reduction:
            pass

        if ess_final > ess_reduction:
            print("Final tempering step, achieved alpha=1.")
            alpha_new = 1
            ess_achieved = ess_final
            iterations = 0
        else:
            m1 = bisect_fct(alpha_last)
            m2 = bisect_fct(1)
            assert np.sign(m1) != np.sign(m2), "bisection function does not have a root"

            rtol = 0.015
            alpha_new, root_result = bisect(
                bisect_fct, alpha_last, 1, rtol=rtol, full_output=True
            )
            iterations = root_result.iterations
            converged = root_result.converged
            ess_achieved = ess_fct(alpha_new)

            if not converged:
                raise RuntimeError(
                    "Bisection algorithm for updating interval did not converge"
                )

        if verbose:
            print(
                "Updating alpha : {:.2f} --> {:.2f}   [ess = {:.2f}, {} iterations]".format(
                    alpha_last, alpha_new, ess_achieved, iterations
                )
            )

        self._interval.temper(alpha_new)
        self._interval.register()

    def _E(self):

        self._E_samples = self._inference.execute()

    def _E_sample(self):

        self._E_samples = self._inference.create_samples()

    def _M(self) -> float:

        P = self._E_samples.detach()
        bs = P.shape[0]
        assert P.ndim == 2

        log_likelihood = self.LogLikelihood(P)

        J = -(log_likelihood) / bs
        J.backward()

        return -J.item()


class FunctionalObjective(AbstractObjective):
    def __init__(
        self,
        model,
        inference,
        log_functional: Callable,
        treat_functional_as_latent=False,
        *,
        dtype=None,
        device=None
    ):

        super().__init__(model, dtype=dtype, device=device)
        self._inference = inference
        self._log_functional = log_functional

        self._dim_kappa = len(self._model.target)

        if treat_functional_as_latent:

            dim = model.rf.kernel.phase_angle_dim + self._dim_kappa

            def logpot(Z):

                kappa = Z[:, 0 : self._dim_kappa]
                Pt = Z[:, self._dim_kappa :]
                return self._J(Pt, kappa)

            self._inference.init(logpot=logpot, dim=dim)
            self._inference.to(dtype=self._dtype, device=self._device)

        else:
            raise DeprecationWarning

        self._treat_functional_as_latent = treat_functional_as_latent

        self._E_samples = None

    @torch.no_grad()
    def sample_phase_angles(self, N):

        R = self._inference._q.sample(N).squeeze()
        assert R.ndim == 2
        return R[:, self._dim_kappa :]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _init(self, *args, **kwargs):
        pass

    def _step(self, *args, **kwargs):
        pass

    def _J(self, Pt: torch.Tensor, kappa: torch.Tensor):

        kappa_mu, kappa_logsigma = self._model.pipeline(
            Pt, mode="transformed_phaseangles"
        )

        logp = DiagonalGaussianLogLikelihood(
            kappa, kappa_mu, 2 * kappa_logsigma, reduce=torch.sum
        )
        assert logp.ndim == 0
        log_prior = self._LogPrior(Pt)
        assert log_prior.ndim == 0
        log_func = self._log_functional(kappa)
        assert log_func.ndim == 1

        return logp + log_prior + log_func.sum()

    def _E(self):

        self._E_samples = self._inference.execute()

    def _M(self) -> float:

        Z = self._E_samples.detach()
        kappa = Z[:, 0 : self._dim_kappa]
        Pt = Z[:, self._dim_kappa :]
        bs = Z.shape[0]
        assert Pt.ndim == 2

        # implementation minimizes the negative ELBO
        J = -self._J(Pt, kappa)

        J.backward()

        return -J.item()


class TargetObjective(FunctionalObjective):
    def __init__(
        self, model, inference, kappa_desired, *, alpha=1, dtype=None, device=None
    ):

        assert len(model.target) == len(kappa_desired)

        class LogTargetFunctional(object):
            def __init__(self, target: torch.Tensor, alpha):
                assert isinstance(target, torch.Tensor)
                assert target.ndim == 1
                self._target = target.view(1, -1)
                assert alpha > 0
                self.alpha = alpha

            def __call__(self, kappa, vars_eval=False, alpha=None):

                if alpha is None:
                    alpha = self.alpha

                if not vars_eval:
                    assert kappa.ndim == 2
                    return -alpha * torch.sum((kappa - self._target) ** 2, 1)
                else:
                    assert kappa.ndim == 3
                    return -alpha * torch.sum((kappa - self._target) ** 2, 2)

        assert isinstance(
            kappa_desired, (torch.Tensor, np.ndarray, list)
        ), "kappa_desired must be array like quantity"
        if not isinstance(kappa_desired, torch.Tensor):
            kappa_desired = torch.tensor(kappa_desired, dtype=dtype, device=device)
        else:
            assert kappa_desired.dtype == dtype
            assert kappa_desired.device == device

        log_functional = LogTargetFunctional(kappa_desired, alpha)
        super().__init__(
            model,
            inference,
            log_functional,
            dtype=dtype,
            device=device,
            treat_functional_as_latent=True,
        )
        self._kappa_desired = kappa_desired

    @torch.no_grad()
    def var(self, Pt: torch.Tensor, N_monte_carlo=512):

        kappa_mean, kappa_logsigma = self._model.pipeline(
            Pt, mode="transformed_phaseangles"
        )

        S = torch.distributions.Normal(kappa_mean, kappa_logsigma).sample(
            (N_monte_carlo,)
        )
        assert S.ndim == 3 and S.shape[0] == N_monte_carlo

        J = self._log_functional(S, vars_eval=True)
        V = torch.var(J, dim=0)
        assert V.ndim == 1 and len(V) == Pt.shape[0]

        return V

    def _assess(self, kappa: torch.Tensor):

        # ugly fix
        return torch.exp(self._log_functional(kappa, alpha=1)).mean().item()

    def set_alpha(self, val):

        # quick fix
        self._log_functional.alpha = val

    def get_alpha(self):

        # quick fix
        return self._log_functional.alpha


class TargetDistribution(object):
    def __init__(self):

        self._samples = None

    def mark_1d(self):

        raise NotImplementedError

    def mark_2d(self):

        raise NotImplementedError


class SampleRepresentationTargetDistribution(TargetDistribution):
    def __init__(self, samples):

        super().__init__()
        self._samples = samples

    def mark_1d(self):

        raise NotImplementedError

    def mark_2d(self):

        raise NotImplementedError


class GaussianMultivariateTargetDistribution(TargetDistribution):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):

        super().__init__()

        assert isinstance(mean, np.ndarray) and mean.ndim == 1
        assert isinstance(cov, np.ndarray) and cov.shape == (len(mean), len(mean))
        assert (
            np.all(np.linalg.eigvals(cov) > 0) and np.linalg.norm(cov - cov.T) < 1e-12
        ), "Coviarance is not SPD"

        self._mean = mean
        self._cov = cov

    def sample(self, N):
        return multivariate_normal.rvs(mean=self._mean, cov=self._cov, size=N)

    def mark_1d(
        self,
        dim: int = None,
        res=20,
        numstd=3,
        xlim: tuple = None,
        color="g",
        width=0.5,
    ):

        if dim is None:
            assert len(self._mean) == 1
            mean = self._mean
            cov = self._cov
        else:
            assert isinstance(dim, int)
            assert 0 <= dim <= len(self._mean)
            mean = self._mean[dim]
            var = self._cov[dim, dim]

        stddev = np.sqrt(var)

        xlim = (
            (mean - numstd * stddev, mean + numstd * stddev) if xlim is None else xlim
        )
        x = np.linspace(xlim[0], xlim[1], res)
        y = singlevariate_normal.pdf(x, loc=mean, scale=stddev)
        plt.plot(x, y, "-", color=color, linewidth=width)

    def mark_2d(
        self,
        dims: tuple = None,
        res=20,
        numstd=3,
        xlim: tuple = None,
        ylim: tuple = None,
        color="g",
        width=0.5,
        N_contours=3,
        c_label_fontsize=None,
        label=None,
        **kwargs
    ):

        if dims is None:
            assert (
                len(self._mean) == 2
            ), "need to specify targeted dimensions if target distribution is not 2D"
            mean = self._mean
            cov = self._cov
        else:
            assert isinstance(dims, list) or isinstance(dims, tuple)
            assert min(dims) > 0 and max(dims) <= len(self._mean)
            mean = self._mean[dims]
            cov = self._cov[dims][:, dims]

        stddev = np.sqrt(np.diag(cov))
        xlim = (
            (mean[0] - numstd * stddev[0], mean[0] + numstd * stddev[0])
            if xlim is None
            else xlim
        )
        ylim = (
            (mean[1] - numstd * stddev[1], mean[1] + numstd * stddev[1])
            if ylim is None
            else ylim
        )
        x = np.linspace(xlim[0], xlim[1], res)
        y = np.linspace(ylim[0], ylim[1], res)
        X, Y = np.meshgrid(x, y)

        Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)
        contours = plt.contour(X, Y, Z, N_contours, colors=color, label=label, **kwargs)
        if c_label_fontsize:
            plt.clabel(contours, inline=True, fontsize=c_label_fontsize)

    def sample(self, N):
        return multivariate_normal.rvs(mean=self._mean, cov=self._cov, size=N)


class KullbackLeiblerLogPotential(torch.nn.Module):

    # auxiliary class for kullback leibler objective
    def __init__(self, kappas: torch.Tensor, wmodel: WrapperModel, objective):

        super().__init__()

        assert isinstance(kappas, torch.Tensor)
        assert kappas.ndim == 2

        self.register_buffer("_kappas", kappas)
        self._wmodel = wmodel
        self._kld_objective = objective

    def collapse(self, Pt, kappas=None):

        N_samples = Pt.shape[0]
        assert Pt.ndim == 3

        Pt_ = Pt.view(-1, Pt.shape[2])

        if kappas is None:
            return Pt_

        assert kappas.ndim == 2
        assert kappas.shape[0] == Pt.shape[1]
        kappa_t_ = kappas.unsqueeze(0).repeat(N_samples, 1, 1).view(-1, kappas.shape[1])

        return Pt_, kappa_t_

    def forward(self, Pt: torch.Tensor, indeces: torch.Tensor = None, verbose=False):

        assert (
            Pt.ndim == 3
        ), "needs to be comprised of (N_samples, batch_dim, dim(phase_angles))"
        K = Pt.shape[0] * Pt.shape[1]

        if indeces is None:
            kappas = self._kappas
        else:
            kappas = self._kappas[indeces]

        Pt, kappas = self.collapse(Pt, kappas)

        assert Pt.ndim == kappas.ndim == 2
        assert Pt.shape[0] == kappas.shape[0]

        loglkl = self._wmodel.LogLikelihood(Pt, kappa=kappas, reduce=torch.sum)
        logprior = -0.5 * torch.sum(Pt ** 2)

        if verbose:
            print("=========")
            print("Logprior: {:.3f}".format(logprior.item() / Pt.shape[0]))
            print("Loglikelihood: {:.3f}".format(loglkl.item() / Pt.shape[0]))
            print("==========")

        J = loglkl + logprior

        # not normalized
        return loglkl + logprior

    def _forward_single(self, Pt, indences, kappa):
        raise NotImplementedError


class KullbackLeiblerObjective(AbstractObjective):
    def __init__(
        self,
        model,
        kappas_t: torch.Tensor,
        inference,
        N_updates: int,
        disseperate=True,
        sampling_strategy="random",
        lr=0.25,
        lr_vi=0.001,
        *,
        device=None,
        dtype=None,
        tempering=True
    ):

        super().__init__(model=model, lr=lr, dtype=dtype, device=device)

        self._tempering = tempering
        self._alpha = None

        assert isinstance(kappas_t, torch.Tensor)
        self._kappas_t = kappas_t.clone().to(dtype=self.dtype, device=self.device)
        self._S = self._kappas_t.shape[0]

        self._disseperate = disseperate

        if not disseperate:
            raise NotImplementedError

        logpot = KullbackLeiblerLogPotential(
            kappas_t.clone().to(dtype=dtype, device=device), model, self
        ).to(dtype=dtype, device=device)
        self._logpot = logpot

        assert sampling_strategy == "random", "currently can only do random sampling"

        self._inference = inference
        self._inference.init(logpot, self.rf.kernel.phase_angle_dim)
        self._inference = self._inference.to(dtype=dtype, device=device)

        self._q_samples = self._inference.create_samples()

        self._N_updates = N_updates
        assert sampling_strategy in [
            "random",
            "cyclic",
            "none",
        ], "sampling strategy {} is not supported".format(sampling_strategy)

        self._sampling_strategy = sampling_strategy

        self._initialized = False

        self.representation = None

        self._cache_tempering = dict()

    @property
    def kappas_target(self):
        return self._kappas_t

    @classmethod
    def FromPhi(
        cls,
        factory,
        phi0: torch.Tensor,
        N_target: int,
        N_updates: int,
        model,
        inference,
        *args,
        **kwargs
    ):

        assert isinstance(factory, SMO.factories.CaseFactory)
        assert (
            isinstance(phi0, list)
            or isinstance(phi0, np.ndarray)
            or isinstance(phi0, torch.Tensor)
        )
        assert (
            N_updates is None
        ), "The N_updates parameter currently is unused, because we always update everything at once"
        assert isinstance(N_target, int) and N_target > 0

        if isinstance(phi0, torch.Tensor):
            assert phi0.ndim == 1
            phi0 = phi0.tolist()

        rf, _ = factory.rf()
        rf.set_phi(phi0)
        hmg = factory.hmg()
        kappas_t = factory.kappa_from_phi(phi0, N=N_target)

        return cls(model, kappas_t, inference, N_updates, *args, **kwargs)

    def _E_sample(self):

        self._q_samples = self._inference.create_samples()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _assess(self, kappa: torch.Tensor):
        return 0

    @torch.no_grad()
    def _init(self, *args, **kwargs):

        if self._tempering:

            X_g = self._model.rf.rsample_batch(self._S)
            kappa_mean, kappa_logsigma = self._model.pipeline_Xg(X_g)
            kappa_samples = reparametrize(kappa_mean, kappa_logsigma)
            self._kappas_i = kappa_samples.detach()
            self._logpot._kappas.data = kappa_samples.clone()
            self._alpha = 0

    @torch.no_grad()
    def _step(self, *args, **kwargs):

        if self._tempering and self._alpha != 1:

            target_stddevs = 5

            Pt = self._q_samples.detach()
            assert (
                Pt.ndim == 3 and Pt.shape[0] == 1
            ), "not implemented for any other cases"

            Pt_collapsed = self._logpot.collapse(Pt)
            kappa_mean, kappa_logsigma = self._model.pipeline(
                Pt_collapsed, mode="transformed_phaseangles"
            )

            def tempered_samples(alpha_):
                return (1 - alpha_) * self._kappas_i + alpha_ * self._kappas_t

            def stddevs(alpha_):

                kappa_samples_active = tempered_samples(alpha_)
                stddevs = torch.abs(
                    (kappa_samples_active - kappa_mean) / torch.exp(kappa_logsigma)
                )
                avg_stddevs = torch.mean(stddevs)
                return avg_stddevs.item()

            def bisect_fct(alpha_):

                return stddevs(alpha_) - target_stddevs

            if stddevs(1) < target_stddevs + 0.5:

                self._alpha = 1
                self._logpot._kappas.data = tempered_samples(1)
                return

            if stddevs(self._alpha) >= (target_stddevs - 0.15):
                return

            m1 = bisect_fct(self._alpha)
            m2 = bisect_fct(1)

            if not (np.sign(m1) != np.sign(m2)):

                warnings.warn(
                    "Failed to find root (m1={:.2f}, m2={:.2f})".format(m1, m2)
                )
                print("--------------- Failed to find root ------------")
                print(
                    "A: {:.2f}   | B : {:.2f}".format(stddevs(self._alpha), stddevs(1))
                )

                if "failed_rootfindings" not in self._cache_tempering:
                    self._cache_tempering["failed_rootfindings"] = 1
                else:
                    self._cache_tempering["failed_rootfindings"] += 1

                    if self._cache_tempering["failed_rootfindings"] > 10:
                        print(
                            "Setting alpha to 1 and abandoning tempering after 10 failed attempts to find a root."
                        )
                        self._alpha = 1
                        self._logpot._kappas.data = tempered_samples(1)
                        return

            else:

                if "failed_rootfindings" in self._cache_tempering:
                    self._cache_tempering["failed_rootfindings"] = 0

                alpha_new, root_result = bisect(
                    bisect_fct, self._alpha, 1, rtol=0.05, full_output=True
                )
                iterations = root_result.iterations
                converged = root_result.converged
                stddevs_achieved = stddevs(alpha_new)

                print(
                    "Setting alpha: {:.2f} ---> {:.2f}".format(self._alpha, alpha_new)
                )
                stddevs_old = stddevs(self._alpha)
                print(
                    "stddevs: {:.2f} ---> {:.2f}".format(stddevs_old, stddevs_achieved)
                )

                if not converged:
                    raise Exception("Tempering for KLD objective failed.")

                self._alpha = alpha_new
                self._logpot._kappas.data = tempered_samples(alpha_new)

    def _E(self, verbose=False):

        if self._disseperate:

            self._q_samples = self._inference.execute()

        else:
            raise NotImplementedError

    def _M(self, verbose=False) -> float:

        Pt = self._q_samples.detach()

        elbo = self._logpot(Pt, verbose=True) / (Pt.shape[0] * Pt.shape[1])

        J = -elbo
        J.backward()

        return -J.item()

    def __repr__(self):

        s = "OBJECTIVE : Kullback Leibler Divergence \n"
        s += "dim(kappa) = {} || N_instances = {} || Subset".format(
            self._kappas_t.shape[1], self._kappas_t.shape[0]
        )
        return s


class TemperedIntervalManager(object):

    # Note: legacy code, but fragments are still used
    def __init__(self, intervals):

        assert (
            isinstance(intervals, np.ndarray) and intervals.shape[1] == 2
        ), "interval needs to be (Mx2) np.ndarray"
        assert np.all(
            intervals[:, 0] < intervals[:, 1]
        ), "left columns needs to contain lower bound, right column upper bound"

        self._t_interval = intervals
        self._i_interval = None
        self._alpha = 0

        self._alpha_hist = list()

        assert np.all(self._t_interval[:, 0] < self._t_interval[:, 1])

    @property
    def bounds(self):
        return self._t_interval

    @property
    def alpha(self):
        return self._alpha

    @property
    def dim(self):
        return self._t_interval.shape[0]

    def register(self):
        self._alpha_hist.append(self._alpha)

    def active_bound(self, alpha=None):

        if alpha is None:
            alpha = self._alpha

        if alpha == 1:
            return self._t_interval
        else:
            return self._i_interval + alpha * (self._t_interval - self._i_interval)

    def temper(self, alpha):
        assert 0 <= alpha <= 1
        self._alpha = alpha

    def linear_step_temper(self, step, max_steps):

        self._alpha = float(step) / max_steps

    def init(self, samples, quantile=0.05):

        mquantiles = (
            torch.quantile(
                samples,
                q=torch.tensor(
                    [quantile, 1 - quantile], dtype=samples.dtype, device=samples.device
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        assert mquantiles.shape[1] == self.dim and mquantiles.shape[0] == 2
        mquantiles = mquantiles.T.copy()
        i_interval_lower = np.minimum(mquantiles[:, 0], self._t_interval[:, 0])
        i_interval_upper = np.maximum(mquantiles[:, 1], self._t_interval[:, 1])
        i_interval = np.column_stack((i_interval_lower, i_interval_upper))
        self._i_interval = i_interval.copy()

    @property
    def _a_interval(self):
        return self._i_interval + self._alpha * (self._t_interval - self._i_interval)

    def lower_bounds(self, dtype, device, alpha=None):

        active_interval = self.active_bound(alpha)
        return torch.tensor(active_interval[:, 0], dtype=dtype, device=device)

    def upper_bounds(self, dtype, device, alpha=None):

        active_interval = self.active_bound(alpha)
        return torch.tensor(active_interval[:, 1], dtype=dtype, device=device)

    def fraction_within_np(self, kappa, active_interval=True, dtype=None, device=None):

        assert isinstance(kappa, np.ndarray)
        kappa = torch.tensor(kappa, dtype=dtype, device=device)

        def _exc(kappa_):
            return self.fraction_within(
                kappa_, active_interval, return_individual=False
            )

        if kappa.ndim == 3:
            results = np.zeros(kappa.shape[0])
            for n, kappa_ in enumerate(kappa):
                results[n] = _exc(kappa_)
            return results
        elif kappa.ndim == 2:
            return _exc(kappa)
        else:
            raise RuntimeError(
                "Shape of kappa array ({}) does not meet requirements".format(
                    kappa.shape
                )
            )

    def fraction_within(self, kappa, active_interval=True, return_individual=True):

        assert kappa.ndim == 2
        assert kappa.shape[1] == self.dim

        interval = self._a_interval if active_interval else self._t_interval
        interval = torch.tensor(interval, dtype=kappa.dtype, device=kappa.device)

        R = (kappa >= interval[:, 0]) & (kappa <= interval[:, 1])
        R_joint = (torch.sum(torch.all(R, 1)) / kappa.shape[0]).item()
        R_single = torch.sum(torch.all(R, 1), 0) / kappa.shape[0]

        if return_individual:
            return R_single.item(), R_joint
        else:
            return R_single.item()

    def __repr__(self):

        if self._i_interval is None:
            return "Uninitialized Interval Manager"

        s = "Target | Active (alpha={:.2f}) \n".format(self._alpha)
        for n in range(self.dim):
            s += "[{:.2f} , {:.2f}] || [{:.2f} , {:.2f}] \n".format(
                self._t_interval[n, 0],
                self._t_interval[n, 1],
                self._a_interval[n, 0],
                self._a_interval[n, 1],
            )

        return s


def ESS(logw: np.ndarray) -> float:

    assert (
        isinstance(logw, np.ndarray) and logw.ndim == 1
    ), "logw needs to be one-dimensional numpy array"

    logw = logw - (np.log(np.sum(np.exp(logw - np.max(logw)))) + np.max(logw))

    n_samples = len(logw)
    ess = (1 / n_samples) * (1 / np.sum(np.exp(2 * logw)))

    # dirty way to deal with edge case
    if ess > 1 and ess < 1.00001:
        ess = 1

    assert 0 <= ess <= 1, "Effective Sample Size outside of admissible bounds"

    return ess
