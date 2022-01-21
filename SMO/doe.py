import numpy as np


class DesignOfExperiment(object):
    def __init__(self, dim, N):

        self._dim = dim
        self._N = N

    @property
    def dim(self):
        return self._dim

    @property
    def N(self):
        if self._N is None:
            raise RuntimeError(
                "Size of Design of Experiment not yet know (not fully initialized"
            )
        else:
            return self._N

    @property
    def deterministic(self):
        return isinstance(self, DeterministicDesignOfExperiment)

    def __len__(self):
        return self._N

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class StochasticDesignOfExperiment(DesignOfExperiment):
    def __init__(self, dim, N):
        super().__init__(dim, N)

    def __iter__(self):

        for n in range(self._N):
            yield self.sample()

    def sample(self):
        raise NotImplementedError


class UniformStochasticDesignOfExperiment(StochasticDesignOfExperiment):
    def __init__(self, dim, vmin, vmax, N, method="lin"):

        super().__init__(dim, N)

        assert len(vmin) == len(vmax) == dim
        if not isinstance(vmin, np.ndarray):
            vmin = np.array(vmin)
        if not isinstance(vmax, np.ndarray):
            vmax = np.array(vmax)
        self._method = method

        self._vmin = vmin
        self._vmax = vmax

    @classmethod
    def Isotropic(cls, dim, vmin, vmax, N, method="lin"):
        return cls(dim, dim * [vmin], dim * [vmax], N, method=method)

    def sample(self):

        if self._method == "lin":
            return np.random.uniform(self._vmin, self._vmax, self._dim)
        elif self._method == "log":
            return 10 ** np.random.uniform(
                np.log10(self._vmin), np.log10(self._vmax), self._dim
            )
        else:
            raise NotImplementedError


class GaussianDesignOfExperiment(StochasticDesignOfExperiment):
    def __init__(self, dim, N):

        super().__init__(dim, N)

    def sample(self):

        return np.random.normal(0, 1, self._dim)


class DeterministicDesignOfExperiment(DesignOfExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        raise NotImplementedError


class SingularPoint(DeterministicDesignOfExperiment):
    def __init__(self, phi, *, N=1):

        super().__init__(len(phi), N)
        if isinstance(phi, list):
            phi = np.array(phi)

        if not isinstance(phi, np.ndarray):
            raise ValueError(
                "Phi needs to be either list or numpy array, not {}".format(type(phi))
            )

        self._phi = [phi for n in range(N)]

    def __iter__(self):

        for phi_ in self._phi:
            yield phi_

    def __getitem__(self, item):
        return self._phi[item]
