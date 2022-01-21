import torch


class DeterministicMetric(object):
    def __init__(self, name):

        self._name = name

    @torch.no_grad()
    def eval(self, X_pred, X_target):

        X_pred = X_pred.view(X_pred.shape[0], -1)
        X_target = X_target.view(X_target.shape[0], -1)
        assert X_pred.shape == X_target.shape
        return self._eval(X_pred, X_target).item()

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def _eval(self, X_pred, X_target):
        raise NotImplementedError

    def __repr__(self):
        return self._name


class MetricEnsemble(object):
    def __init__(self, metrics):

        for metric in metrics:
            assert isinstance(metric, DeterministicMetric)

        self._metrics = metrics

    def keys(self):
        return [str(metric) for metric in self._metrics]

    def eval(self, X_pred, X_target):

        results = dict()

        for metric in self._metrics:
            results[str(metric)] = metric.eval(X_pred, X_target)

        return results


class MSE(DeterministicMetric):
    def __init__(self):

        super().__init__("Mean Squared Error")

    def _eval(self, X_pred, X_target):

        return torch.mean((X_pred - X_target) ** 2)


class CoefficientOfDetermination(DeterministicMetric):
    def __init__(self, global_average=True):

        super().__init__("Coefficient of Determination")
        self._global_average = global_average

    def _eval(self, X_pred, X_target):

        y_pred = X_pred
        y = X_target

        if self._global_average:
            # this is the implementation from torch.ignite
            e = torch.sum((y - y_pred) ** 2) / torch.sum((y - y.mean()) ** 2)
            return 1 - e

        else:
            # component-wise mean
            assert y_pred.shape[0] > 0
            e = torch.sum((y - y_pred) ** 2, 0) / torch.sum((y - y.mean(0)) ** 2, 0)
            return (1 - e).mean()


def IndividualR2(Y_pred: torch.Tensor, Y_target: torch.Tensor) -> list:

    y = Y_target
    e = torch.sum((y - Y_pred) ** 2, 0) / torch.sum((y - y.mean(0)) ** 2, 0)
    return e.tolist()
