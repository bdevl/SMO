import torch
import numpy as np


class LearningScheduleWrapper(object):
    def __init__(self, create_scheduler, disable=False):

        self._schedulers = dict()
        self._optimizers = dict()
        self._disable = disable
        self._counter = dict()

        self._create_scheduler = create_scheduler  #

    def lock(self):
        self._disable = True

    def unlock(self):
        self._disable = False

    @classmethod
    def stepLR(cls, step_size, factor=0.1):
        def f(optimizer):
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=factor, last_epoch=-1
            )

        return cls(f)

    @classmethod
    def ReduceLROnPlateau(
        cls, patience, threshold=1e-3, factor=0.1, min_lr=1e-3, verbose=True, mode="max"
    ):

        assert factor < 1

        def f(optimizer):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=mode,
                patience=patience,
                threshold=threshold,
                factor=factor,
                min_lr=min_lr,
                verbose=verbose,
            )

        return cls(f)

    @classmethod
    def MultiStepLR(cls, milestones, factor, last_epoch=-1):

        assert factor < 1

        def f(optimizer):
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=factor, last_epoch=last_epoch
            )

        return cls(f)

    @classmethod
    def Dummy(cls):
        def f(optimizer):
            return None

        wrapper = cls(f)
        wrapper.lock()
        return wrapper

    def set_learning_rate_manually(self, id, lr):

        for param_group in self._optimizers[id].param_groups:
            param_group["lr"] = lr

    def register_optimizer(self, optimizer, id="default"):

        self._schedulers[id] = self._create_scheduler(optimizer)
        self._optimizers[id] = optimizer
        self._counter[id] = 0

    def step(self, id="default", N=None, N_max=None, interval=1, metric=None):

        if interval is None:
            # legacy fix
            interval = 1

        if id not in self._optimizers:
            raise ValueError(
                'LearningScheduleWrapper does not have "{}" optimizer registered'.format(
                    id
                )
            )

        self._counter[id] += 1

        if self._disable:
            return

        if np.mod(self._counter[id], interval) != 0:
            return

        if isinstance(self._schedulers[id], torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires metric")
            self._schedulers[id].step(metric)
        else:
            self._schedulers[id].step()


def utiliy_convert_milestones(init_lr, milestones, factor, verbose=True):

    learning_rates = [init_lr] + [
        init_lr * (factor ** n) for n in range(len(milestones))
    ]
    milestones = [0] + milestones

    string = " ============================== \n"
    for n, (milestone, lr) in enumerate(zip(milestones, learning_rates)):

        if n < len(milestones) - 1:
            upper = milestones[n + 1]
        else:
            upper = "inf"

        string += "For epoch {0} - {1} ------ lr: {2:.4f} \n".format(
            milestone, upper, lr
        )
    string += " ============================== \n"

    if verbose:
        print(string)

    return learning_rates, string
