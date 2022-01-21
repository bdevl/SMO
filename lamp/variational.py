from lamp.modules import BaseModule
import torch
import torch.distributions


class VariationalApproximation(BaseModule):
    def __init__(self, dim, N):

        super().__init__()

        self._dim = dim
        self._N = N

        self._data_hash = None

    def _create_pt_distribution(self):

        raise NotImplementedError

    def _create_pt_distribution_indexed(self, indeces):

        raise NotImplementedError

    def LogEvaluate(self, X, indeces=None):

        if indeces is None:
            q = self._create_pt_distribution()
        else:
            q = self._create_pt_distribution_indexed(indeces)

        return q.log_prob(X)

    def register_data(self, X=None):

        if X is None:
            return

        if self._data_hash is None:
            self._data_hash = hash(X)
        else:
            raise RuntimeError(
                "Trying to register data to VariationalApproximation for which a datahash already exists"
            )

    def check_data(self, X, throw_exception=False):

        if self._data_hash is None:
            raise RuntimeError(
                "No data has been registered for the VariationalApproximation"
            )

        r = self._data_hash == hash(X)

        if throw_exception and not r:
            raise RuntimeError("The data hash does not match the tensor X")

        return r

    @property
    def N(self):
        return self._N

    @property
    def dim(self):
        return self._dim

    def sample(self, *args, **kwargs):

        return self.rsample(*args, **kwargs)

    def rsample(self, num_samples=1):

        q = self._create_pt_distribution()
        return q.rsample(num_samples=num_samples)

    def sample_batch_component(self, index, num_samples):

        try:
            q = self._create_pt_distribution_indexed(index)
        except NotImplementedError as e:
            raise NotImplementedError(
                "sample_batch_component() needs to implemented in child-class, because _create_pt_distribution_indexed() is not available."
            )

        return q.rsample(num_samples=num_samples)

    def _register_data(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class BaseMultivariateNormal(VariationalApproximation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def logsigma(self):
        raise NotImplementedError

    def KLD(self):
        raise NotImplementedError


class DiagonalMultivariateNormal(VariationalApproximation):
    def __init__(self, dim, N):

        super().__init__(dim=dim, N=N)

        self._logsigma = torch.nn.Parameter(torch.zeros(N, dim, requires_grad=True))
        self._mean = torch.nn.Parameter(torch.zeros(N, dim, requires_grad=True))

    @property
    def mean(self):
        return self._mean

    @property
    def logsigma(self):
        return self._logsigma

    def _create_pt_distribution(self):

        return torch.distributions.Normal(
            loc=self._mean, scale=torch.exp(self._logsigma)
        )

    def _create_pt_distribution_indexed(self, index):

        return torch.distributions.Normal(
            loc=self._mean[index], scale=torch.exp(self._logsigma[index])
        )

    def rsample(self, num_samples=1):

        q = self._create_pt_distribution()

        if num_samples == 1:
            return q.rsample()
        else:
            return q.rsample((num_samples,))


class LowRankMultivariateNormal(VariationalApproximation):
    def __init__(self, dim, N, M, sfactor=1):

        super().__init__(dim=dim, N=N)

        batch_shape = N

        self._loc = torch.nn.Parameter(
            torch.zeros((batch_shape, dim), requires_grad=True)
        )
        self._cov_factor = torch.nn.Parameter(
            sfactor * torch.randn((batch_shape, dim, M), requires_grad=True)
        )
        self._cov_log_diag = torch.nn.Parameter(
            torch.zeros((batch_shape, dim), requires_grad=True)
        )

        self._M = M

    @property
    def loc(self):
        return self._loc.data

    @loc.setter
    def loc(self, value):

        assert isinstance(value, torch.Tensor)
        assert value.shape == self._loc.shape
        assert value.dtype == self._loc.dtype
        assert value.device == self._locl.device
        self._loc.data = value

    @property
    def M(self):
        return self._M

    def _create_pt_distribution(self):

        return torch.distributions.LowRankMultivariateNormal(
            loc=self._loc,
            cov_factor=self._cov_factor,
            cov_diag=torch.exp(self._cov_log_diag),
        )

    def _create_pt_distribution_indexed(self, indeces: torch.Tensor):

        return torch.distributions.LowRankMultivariateNormal(
            loc=self._loc[indeces],
            cov_factor=self._cov_factor[indeces],
            cov_diag=torch.exp(self._cov_log_diag[indeces]),
        )

    def rsample(self, num_samples=1, indeces=None):

        if indeces is None:
            q = self._create_pt_distribution()
        else:
            q = self._create_pt_distribution_indexed(indeces)

        return q.rsample((num_samples,))
