from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal, kde


def softmax(x: np.ndarray, dim=1) -> np.ndarray:

    if x.ndim == 1:
        return torch.nn.Softmax(0)(torch.tensor(K)).detach().cpu().numpy()

    return torch.nn.Softmax(dim)(torch.tensor(K)).detach().cpu().numpy()


def DiscriminativeModelObjectiveFunction(factory, interval, model, N_samples=None):

    if N_samples is None:
        rf_samples, doe = factory.doe_reference_samples(return_doe=True)
    else:
        rf_samples, doe = factory.doe_reference_samples(
            return_doe=True, N_samples=N_samples
        )

    assert doe.dim == 2, "this function is intended for dim(phi) = 2 only"
    assert doe.deterministic, "only applicable for deterministic DOEs"

    objfct = list()

    with torch.no_grad():
        for sample in rf_samples:
            kappa_samples = model.propagate(
                sample.to(dtype=model.dtype, device=model.device)
            )
            objfct.append(
                interval.fraction_within(
                    kappa_samples, active_interval=False, return_individual=False
                )
            )

    ind_max = np.argmax(np.array(objfct))

    info = dict()
    info["phi_max"] = doe[ind_max]
    info["objfct_max"] = objfct[ind_max]

    XX, YY = doe.grid

    return XX, YY, np.array(objfct).reshape(XX.shape), info


class DataTransformation(torch.nn.Module):
    def __init__(self, nx, differentiable, *, image_channel=True):

        super(DataTransformation, self).__init__()
        self._nx = nx
        self._image_channel = image_channel
        self._differentiable = differentiable

    @property
    def differentiable(self):
        return self._differentiable

    def _reshape(self, x):

        if self._image_channel:
            return x.view(-1, 1, self._nx, self._nx)
        else:
            return x

    def forward(self):
        raise NotImplementedError


class IdentityDataTransform(DataTransformation):
    def __init__(self, nx, **kwargs):
        super(IdentityDataTransform, self).__init__(nx, True, **kwargs)

    def forward(self, x):
        return self._reshape(x)


class ExponentialDataTransform(DataTransformation):
    def __init__(self, nx, **kwargs):
        super(ExponentialDataTransform, self).__init__(nx, True, **kwargs)

    def forward(self, x):

        return self._reshape(torch.exp(x))

    def __repr__(self):
        return "Exponential Transformation"


class BinarizeDataTransform(DataTransformation):
    def __init__(self, nx, cutoff, phase_high, phase_low, **kwargs):
        super(BinarizeDataTransform, self).__init__(nx, False, **kwargs)

        self._cutoff = cutoff
        self._phase_high = phase_high
        self._phase_low = phase_low

    def forward(self, x):

        x_b = torch.full_like(x, self._phase_low)
        x_b[x >= self._cutoff] = self._phase_high
        return self._reshape(x_b)

    def __repr__(self):

        return "Binarization transformation ( {} | {} )".format(
            self._phase_low, self._phase_high
        )


class HyperbolicDataTransform(DataTransformation):
    def __init__(self, nx, eps=1, cutoff=0, **kwargs):

        super().__init__(nx, True, **kwargs)
        assert eps >= 0
        self._tanh = torch.nn.Tanh()
        self._eps = eps
        self._cutoff = cutoff

    def forward(self, x):

        x = self._reshape(x)

        if self._eps > 0:
            # note: eps=0 is assumed to encode identity mapping
            x = self._tanh(self._eps * (x - self._cutoff))

        return x

    def __str__(self):

        if self._eps > 0:
            s = "Data transformation: Soft thresholding with eps = {}".format(self._eps)
        else:
            s = "Data transformation: identity map"

        return s


class SigmoidDataTransform(DataTransformation):
    def __init__(self, nx, eps=1, **kwargs):

        super.__init__(nx, True, **kwargs)
        assert eps >= 0
        self._sigmoid = torch.nn.Sigmoid()
        self._eps = eps

    def forward(self, x):

        return self._reshape(-1 + 2 * self._sigmoid(self._eps * x))

    def __str__(self):
        return "Data Transformation : Sigmoid / Logistic -> [-1, +1]"


def binarize(Z, cutoff, phase_high=None, phase_low=None, logtransform=True):

    if not logtransform:
        raise NotImplementedError

    if phase_high is None and phase_low is None:
        return (Z > cutoff).float()
    else:
        high_phase_ind = Z > cutoff
        R = torch.full_like(Z, np.log(phase_low))
        R[high_phase_ind] = np.log(phase_high)
        return R


class Gaussian2D(object):
    def __init__(self, mean, cov):

        self._mean = mean
        self._cov = cov

    def plot(self, res=20, numstd=3):

        stddev = np.sqrt(np.diag(self._cov))
        xlim = [self._mean[0] - numstd * stddev[0], self._mean[0] + numstd * stddev[0]]
        ylim = [self._mean[1] - numstd * stddev[1], self._mean[1] + numstd * stddev[1]]
        x = np.linspace(xlim[0], xlim[1], res)
        y = np.linspace(ylim[0], ylim[1], res)
        X, Y = np.meshgrid(x, y)
        X = X
        Y = Y

        Z = multivariate_normal.pdf(np.dstack((X, Y)), mean=self._mean, cov=self._cov)
        contours = plt.contour(X, Y, Z, 3, colors="g")
        plt.clabel(contours, inline=True, fontsize=8)
        ax = plt.gca()

    def sample(self, N):
        return multivariate_normal.rvs(mean=self._mean, cov=self._cov, size=N)


def plotdensity2d(k1, k2, nbins=40, k1r=None, k2r=None):

    kmin = [k1.min(), k2.min()]
    kmax = [k1.max(), k2.max()]

    kde_ = kde.gaussian_kde([k1, k2])
    xi, yi = np.mgrid[kmin[0] : kmax[0] : nbins * 1j, kmin[1] : kmax[1] : nbins * 1j]
    zi = kde_(np.vstack([xi.flatten(), yi.flatten()]))
    h1 = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", cmap=plt.cm.magma)
    plt.colorbar(h1)

    return h1


def cuda_available(self):
    return torch.cuda.is_available()


def data_to_be_used(cargs):

    return [
        cargs["N_training_init"] + n * cargs["N_add"]
        for n in range(cargs["N_data_acquisitions"] + 1)
    ]


def substitute_defaults(default_values, provided_args):

    for key, value in default_values.items():
        if key not in provided_args:
            provided_args[key] = deepcopy(value)
