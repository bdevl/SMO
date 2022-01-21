import numpy as np
import torch
from lamp.variational import LowRankMultivariateNormal


class Inference(object):
    def __init__(self):
        self._logpot = None
        self._dim = None

        self._counter = 0
        self._ilog = list()

        self._device = torch.device("cpu")
        self._dtype = torch.float32

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def execute(self, *args, **kwargs):

        self._counter += 1
        self._ilog.append(dict())

        return self._execute(*args, **kwargs)

    def to(self, dtype=None, device=None):
        raise NotImplementedError

    def __getitem__(self, item):
        return self._ilog[item]

    def init(self, logpot, dim, *args, **kwargs):

        self._logpot = logpot
        self._dim = dim
        self._init(*args, **kwargs)

    def init_target(self, target, *args, **kwargs):
        self._logpot = target.LogEvaluate
        self._dim = target.dim
        self._init(*args, **kwargs)

    def _init(self, *args, **kwargs):
        raise NotImplementedError

    def _execute(self, *args, **kwargs):
        raise NotImplementedError

    def _log(self, *args, **kwargs):
        raise NotImplementedError


class LowRankVariationalInference(Inference):
    def __init__(
        self,
        N_iterations,
        M_lowrank,
        N_samples=1,
        lr=1e-3,
        batch_size=None,
        N_samples_em=64,
        sub_sampling_size=None,
    ):

        super().__init__()
        self._N_iterations = N_iterations
        self._M_lowrank = M_lowrank
        self._N_samples = N_samples
        self._N_samples_em = N_samples_em
        self._batch_size = batch_size
        self._sub_sampling_size = sub_sampling_size
        self._lr = lr
        self._q = None
        self._optim = None
        self._elbo_dedicated = None

    def elbo(self, N_avg=1):

        if self._elbo_dedicated is not None:
            return self._elbo_dedicated

        if "elbo" not in self._ilog[-1]:
            raise RuntimeError

        if N_avg == 1:
            return self._ilog[-1]["elbo"][-1]
        else:
            assert len(self._ilog[-1]["elbo"]) > N_avg
            return np.array(self._ilog[-1]["elbo"])[-N_avg:].mean()

    def elbo_precise(self, N):

        assert N > 1

        with torch.no_grad():

            x = self._q.rsample(N)

            if self._batch_size is None:

                x = x.squeeze(1)
                assert x.shape[0] == N
                assert x.ndim == 2
                elbo = self._logpot(x) - torch.sum(self._q.LogEvaluate(x))
                elbo = elbo / N
                return elbo.item()

            else:

                elbo = self._logpot(x) - torch.sum(self._q.LogEvaluate(x))
                elbo = elbo / (x.shape[0] * x.shape[1])
                return elbo.item()

    def to(self, dtype=None, device=None):

        assert isinstance(dtype, torch.dtype) or dtype is None
        assert isinstance(device, torch.device) or device is None

        if dtype is not None:
            self._dtype = dtype

        if device is not None:
            self._device = device

        self._q.to(dtype=dtype, device=device)

        return self

    def _init(self):

        batch_size_dim = 1 if self._batch_size is None else self._batch_size
        self._q = LowRankMultivariateNormal(
            self._dim, batch_size_dim, self._M_lowrank, sfactor=0.1
        ).to(dtype=self.dtype, device=self.device)

        self._optim = torch.optim.Adam(self._q.parameters(), lr=self._lr)

    def _log(self, elbo_hist):

        self._ilog[-1]["elbo"] = elbo_hist

    @torch.no_grad()
    def create_samples(self, N_samples=None, indeces=None):

        if N_samples is None:
            N_samples = self._N_samples_em

        X = self._q.rsample(N_samples, indeces=indeces)

        assert X.shape[0] == N_samples
        assert X.shape[2] == self._dim

        if self._batch_size is None:
            assert X.shape[1] == 1
        else:
            assert X.shape[1] == self._batch_size

        if self._batch_size is None:
            X.squeeze_(1)

        return X.detach()

    def _execute(self):

        elbo_hist = list()

        for n in range(self._N_iterations):

            if self._sub_sampling_size is not None:

                raise DeprecationWarning

            else:

                x = self._q.rsample(self._N_samples)
                assert x.ndim == 3

                N_normalize = x.shape[0] * x.shape[1]
                if self._batch_size is None:
                    x = x.squeeze(1)

                elbo = self._logpot(x) - torch.sum(self._q.LogEvaluate(x))

            elbo = elbo / N_normalize
            J = -elbo
            self._optim.zero_grad(True)
            J.backward()
            self._optim.step()
            elbo_hist.append(-J.item())

        self._log(elbo_hist)

        return self.create_samples()
