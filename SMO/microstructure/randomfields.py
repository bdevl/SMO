import torch
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import warnings
import lamp.modules


class RandomFieldParameters(object):
    def __init__(self, kernel):

        self._kernel = kernel

    @property
    def dim(self):
        return self._kernel.pdim

    def clamp(self):
        raise NotImplementedError

    def __repr__(self):
        return "Random Field Parameters | process parameters | dim = {}".format(
            self.dim
        )


class UnboundedRandomFieldParameters(RandomFieldParameters):
    def __init__(self, kernel):

        super().__init__(kernel)

    def clamp(self):
        pass


class SoftmaxRandomFieldParameters(RandomFieldParameters):
    def __init__(self, kernel):

        super().__init__(kernel)
        self._max_discrepancy = 10

    def clamp(self):

        params = self._kernel.phi
        diff = params.data.max().item() - params.data.min().item()

        if diff > self._max_discrepancy:
            f = diff / self._max_discrepancy
            params.data = params.data / f


class IntervalRandomFieldParameters(RandomFieldParameters):
    def __init__(self, kernel, bounds=None):

        super().__init__(kernel)

        self._kernel = kernel
        assert isinstance(bounds, np.ndarray)
        assert bounds.shape[0] == self.dim
        assert bounds.shape[1] == 2

        self._bounds = bounds

    def clamp(self):

        params = self._kernel.phi

        assert hasattr(
            self._kernel, "inverse_transform"
        ), "The chosen kernel is not compatible with the random field parameter constraint."

        upperbound = self._kernel.inverse_transform(
            torch.tensor(self._bounds[:, 1], dtype=params.dtype, device=params.device)
        )
        lowerbound = self._kernel.inverse_transform(
            torch.tensor(self._bounds[:, 0], dtype=params.dtype, device=params.device)
        )

        data = params.data.clone()
        data = torch.where(params < lowerbound, lowerbound, data)
        data = torch.where(params > upperbound, upperbound, data)

        params.data = data


class PhaseAngleNormalTransform(torch.nn.Module):
    def __init__(self, dtype, device):

        super().__init__()
        self._dtype = dtype
        self._device = device
        self.register_buffer(
            "_2pi", torch.tensor(2 * np.pi, dtype=dtype, device=device)
        )
        self.register_buffer(
            "_sqrt2", torch.tensor(np.sqrt(2), dtype=dtype, device=device)
        )

    def forward(self, Z):

        # with Z ~ N(0,1) a vector (will be of dim 2*Nw**2)
        return 0.5 * (1 + torch.erf(Z / self._sqrt2)) * self._2pi


class Kernel(torch.nn.Module):
    def __init__(self, dim, pdim, w_max, N):

        super().__init__()

        self._dim = dim
        self._phi = None
        self._pdim = pdim
        self._N = dict()
        self._w_max = dict()
        self._WX = None

        self._setup(N, w_max)

    @property
    def phi(self):

        return self._phi

    @phi.setter
    def phi(self, val):
        assert isinstance(val, torch.Tensor)
        assert len(val) == self.pdim
        self._phi.data = val

    # enable construction of new kernels hierarchically. not fully implemented / used
    def __add__(self, other):
        return AggregateKernel(self, other, KernelAddition())

    def __sub__(self, other):
        return AggregateKernel(self, other, KernelSubtraction())

    def __mul__(self, other):
        return AggregateKernel(self, other, KernelMultiplication())

    @property
    def dtype(self):
        return self._phi.dtype

    @property
    def device(self):
        return self._phi.device

    @property
    def pdim(self):
        return self._pdim

    @property
    def dim(self):
        return self._dim

    @property
    def phase_angle_dim(self):
        return 2 * np.prod(np.array(list(self._N.values())))

    @property
    def dim_phase_angles(self):
        # lazy fix instead of refactoring
        return self.phase_angle_dim

    def dw(self, dir):

        return self._w_max[dir] / self._N[dir]

    def _setup(self, N, w_max):

        # prepare for batched execution of the kernel at the pre-specified grid
        if self._dim == 1:
            raise NotImplementedError
        elif self._dim == 2:
            if isinstance(N, list):
                self._N["x"] = N[0]
                self._N["y"] = N[1]
            elif isinstance(N, int):
                self._N["x"] = N
                self._N["y"] = N
            else:
                raise RuntimeError(
                    "Expects resolution information of frequency domain to be either given by list or integer"
                )

            if isinstance(w_max, float):
                self._w_max["x"] = self._w_max["y"] = w_max
            elif isinstance(w_max, list):
                self._w_max["x"], self._w_max["y"] = w_max[0], w_max[1]

    def SDF(self, *args, **kwargs):

        raise NotImplementedError

    def PlotFrequencyDomain(self, cmap="magma", add_title="", colorbar=False):

        if self._dim == 1:
            raise NotImplementedError
        elif self._dim == 2:

            w1 = torch.linspace(
                0, self._w_max["x"], self._N["x"], dtype=self.dtype, device=self.device
            )
            w2 = torch.linspace(
                0, self._w_max["y"], self._N["y"], dtype=self.dtype, device=self.device
            )
            Z = self.SDF(w1, w2)
            W2, W1 = torch.meshgrid(w2, w1)
            h = plt.contourf(
                W2.detach().cpu().numpy(),
                W1.detach().cpu().numpy(),
                Z.detach().cpu().numpy().T,
                cmap="magma",
            )
            if colorbar:
                plt.colorbar(h)
            plt.title("Spectral Density Function " + add_title)
            plt.xlabel(r"Frequency $w_x$")
            plt.ylabel(r"Frequency $w_y$")

            return W1, W2, Z

        elif self._dim == 3:
            raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class GaussianGridKernel(Kernel):
    def __init__(self, dim, w_max, N, N_g, sigma_w, init_uniform=False):

        pdim = N_g ** dim
        assert dim == 2, "currently only support the 2D case"
        super().__init__(dim, pdim, w_max, N)

        assert sigma_w > 0
        self._N_g = N_g
        self._sigma_w = sigma_w

        self._phi = torch.nn.Parameter(torch.randn(pdim))

        if init_uniform:
            self._phi.data = torch.zeros(pdim)

        self._softmax = torch.nn.Softmax(0)

        self.register_buffer(
            "_loc",
            torch.stack(
                torch.meshgrid(
                    torch.linspace(0, w_max, N_g), torch.linspace(0, w_max, N_g)
                ),
                2,
            ).view(1, N_g, N_g, 1, 1, dim),
        )

    def _transform(self):
        return self._softmax(self._phi)

    def set_phi(self, phi: Union[torch.Tensor, list, np.ndarray]):

        assert isinstance(phi, list) or (
            (isinstance(phi, torch.Tensor) or isinstance(phi, np.ndarray))
            and phi.ndim == 1
        ), "Invalid phi type, {}".format(type(phi))
        if isinstance(phi, np.ndarray):
            phi = phi.copy()
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, dtype=self.phi.dtype, device=self.phi.device)

        self.phi = phi

    def get_phi(self) -> list:

        assert self._phi.ndim == 1
        return self.phi.tolist().copy()

    def SDF(self, w1, w2):

        N_g = self._N_g
        N_w = len(w1)
        bs = 1
        q_ = torch.distributions.Normal(self._loc, scale=self._sigma_w)
        q = torch.distributions.Independent(q_, 1)
        v = torch.stack(torch.meshgrid(w1, w1), 2).view(1, 1, 1, N_w, N_w, 2)
        R = torch.exp(q.log_prob(v))
        R = R * self._transform().view(bs, N_g, N_g, 1, 1)
        R = torch.sum(R, dim=(1, 2)).squeeze()

        return R


class DifferentiableGaussianRandomField(torch.nn.Module):
    def __init__(
        self,
        ns,
        kernel,
        dim,
        w_max=None,
        Nw=None,
        dtype=None,
        device=None,
        precomputation=True,
    ):

        super().__init__()

        assert device is not None and dtype is not None
        if w_max is not None or Nw is not None:
            warnings.warn(
                "Passing any information about the discretization in the phase space to DifferentiableGaussianRandomField is deprecated"
            )

        self._kernel = kernel

        assert dim == 2, "can only deal with spatial dimension 2"
        self._dim = dim
        self._domain = None
        self._grid = None
        self._dim_param = None

        assert ns > 1 and isinstance(
            ns, int
        ), "spatial resolution must be positive integer larger than one."
        self._ns1 = ns
        self._ns2 = ns
        assert self._ns1 == self._ns2, "implementation assumes ns1=ns2."

        # cached basis, if precomputation is set to true
        self._precumptation = precomputation

        self._pant = PhaseAngleNormalTransform(dtype=dtype, device=device)

        self._setup()

        self.register_buffer(
            "_sqrt2", torch.tensor([np.sqrt(2)], dtype=dtype, device=device)
        )
        self.register_buffer("_pi", torch.tensor([np.pi], dtype=dtype, device=device))

    @property
    def _Nw(self):

        assert self._kernel._N["x"] == self._kernel._N["y"]
        return self._kernel._N["x"]

    @property
    def _w_max(self):

        assert self._kernel._w_max["x"] == self._kernel._w_max["y"]
        return self._kernel._w_max["x"]

    @property
    def ns(self):
        return (self._ns2, self._ns1)

    @property
    def kernel(self):
        return self._kernel

    @property
    def dtype(self):
        return self._kernel.dtype

    @property
    def device(self):
        return self._kernel.device

    def set_phi(self, phi):
        self._kernel.set_phi(phi)

    def get_phi(self):
        return self._kernel.get_phi()

    def _setup(self):

        if self._dim == 1:
            raise NotImplementedError
        elif self._dim == 2:

            self.register_buffer("_vs1", torch.linspace(0, 1, self._ns1))
            self.register_buffer("_vs2", torch.linspace(0, 1, self._ns2))

            self.register_buffer("_vw1", torch.linspace(0, self._w_max, self._Nw))
            self.register_buffer("_vw2", torch.linspace(0, self._w_max, self._Nw))

            if self._precumptation:

                B1, B2 = self._basis(ApplyCos=False)
                self.register_buffer("_B1", B1)
                self.register_buffer("_B2", B2)

                dwf = self._kernel.dw("x") * self._kernel.dw("y")
                mask = 2 * torch.ones(self._Nw, self._Nw)
                mask[-1, 0] = 0.25
                mask[-1, 1:] = 1
                mask[0:-1, 0] = 1
                mask = mask * dwf
                self.register_buffer("_mask", mask)

            else:
                self._B1 = None
                self._B2 = None

        else:
            raise RuntimeError

    def rsample_transform(self, theta_hat):

        theta = self._pant(theta_hat)
        assert theta.shape == theta_hat.shape
        return self.rsample(theta=theta, acknowledge_untransformed_phase_angles=True)

    def rsample(self, theta=None, *, acknowledge_untransformed_phase_angles=False):

        if not acknowledge_untransformed_phase_angles and theta is not None:
            raise RuntimeError(
                "check whether the instance calling this is doing so correctly (U[0,2\pi])"
            )

        if theta is not None and theta.ndim > 1:

            assert theta.ndim == 2, "needs to be batch_dim x phase_angle_dim"
            bs = theta.shape[0]
            X = torch.zeros(
                bs, self._ns2, self._ns1, dtype=self.dtype, device=self.device
            )

            for n in range(bs):
                theta_ = theta[n]
                X[n, :, :] = self._rsample(theta_)

            return X
        else:
            return self._rsample(theta)

    def _rsample(self, theta: torch.Tensor = None):

        if not self._precumptation:
            raise NotImplementedError

        if theta is not None:
            assert isinstance(theta, torch.Tensor)
            assert (
                theta.ndim == 1
            ), "the internal _rsample function expects a one-dimensional tensor (transformed phase angles)"
            assert len(theta) == self._kernel.dim_phase_angles

        A1 = torch.sqrt(self._mask * self._kernel.SDF(self._vw1, self._vw2))
        A2 = torch.sqrt(self._mask * self._kernel.SDF(self._vw1, -self._vw2))

        if theta is None:
            P1 = (
                torch.rand(self._Nw, self._Nw, dtype=A1.dtype, device=A1.device).view(
                    1, 1, 1, self._Nw, self._Nw
                )
                * 2
                * self._pi
            )
            P2 = (
                torch.rand(self._Nw, self._Nw, dtype=A1.dtype, device=A1.device).view(
                    1, 1, 1, self._Nw, self._Nw
                )
                * 2
                * self._pi
            )
        else:
            Pv = torch.split(theta, self._Nw ** 2)
            P1 = Pv[0].view(self._Nw, self._Nw)
            P2 = Pv[1].view(self._Nw, self._Nw)

        S = self._sqrt2 * (
            A1 * torch.cos(self._B1 + P1) + A2 * torch.cos(self._B2 + P2)
        )

        return S.sum(4).sum(3).squeeze()

    def rsample_batch(self, batch_size):

        # just loop due to memory constraints
        X = torch.zeros(
            batch_size, self._ns2, self._ns1, dtype=self.dtype, device=self.device
        )

        for n in range(batch_size):
            X[n, :, :] = self.rsample()

        return X

    def _basis(self, ApplyCos=False):

        if self._dim == 1:
            raise NotImplementedError
        elif self._dim == 2:

            B1 = self._vs1.view(1, 1, -1, 1, 1) * self._vw1.view(
                1, 1, 1, 1, -1
            ) + self._vs2.view(1, -1, 1, 1, 1) * self._vw2.view(1, 1, 1, -1, 1)
            B2 = self._vs1.view(1, 1, -1, 1, 1) * self._vw1.view(
                1, 1, 1, 1, -1
            ) - self._vs2.view(1, -1, 1, 1, 1) * self._vw2.view(1, 1, 1, -1, 1)

            if ApplyCos:

                B1 = torch.cos(B1)
                B2 = torch.cos(B2)

            return B1, B2

        else:
            raise NotImplementedError(
                "can only deal with random field in (spatial) dimension 2"
            )

    def illustrate_basis(self, basis_fct_index: int):

        B1, B2 = self._basis()

        ind = np.unravel_index(basis_fct_index, (B1.shape[3], B1.shape[4]))
        return (
            B1[:, :, :, ind[0], ind[1]].squeeze().detach().cpu().numpy(),
            B2[:, :, :, ind[0], ind[1]].squeeze().detach().cpu().numpy(),
        )

    def sample_transformed_phase_angles(self, N: int) -> torch.Tensor:

        return torch.randn(
            N, self.kernel.dim_phase_angles, dtype=self.dtype, device=self.device
        )
