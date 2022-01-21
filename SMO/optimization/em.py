import time
import numpy as np
import torch
import matplotlib.pyplot as plt


class ExpectationMaximization(object):
    def __init__(self, objective, active_learner=None):

        if active_learner is not None:
            raise DeprecationWarning

        self._objective = objective
        self._logdicts = list()
        self._counter = 0

        self._elbos = list()
        self._ovals = list()

    @property
    def objective(self):
        return self._objective

    @property
    def phi_trajectory(self) -> np.ndarray:

        return np.array([self._logdicts[n]["phi"] for n in range(len(self._logdicts))])

    @property
    def rf(self):
        return self._objective.rf

    def _log(self, init=False):

        phi = np.asarray(self.rf.kernel.get_phi())
        self._logdicts.append(dict())
        self._logdicts[-1]["phi"] = phi

    def run_active_learning(
        self,
        ConvergenceCriteria,
        ActiveLearner,
        N_elbo_estimate=256,
        N_objective_estimate=256,
        callback=None,
        log=None,
        M_steps=1,
    ):

        # note: assumes trained model (initally)
        phis = list()
        objective_fct = list()

        counter = 0
        while ActiveLearner:

            counter += 1

            if counter > 1:
                ActiveLearner()
                self._objective.wmodel.autotrain()

            self._run(
                ConvergenceCriteria, N_elbo_estimate=N_elbo_estimate, M_steps=M_steps
            )

            # not essential, just monitoring. awkward approach, but gets the job done.
            _, X_cnn, kappas = ActiveLearner._homogenize(
                torch.randn(
                    N_objective_estimate,
                    self.rf.kernel.dim_phase_angles,
                    dtype=self._objective.dtype,
                    device=self._objective.device,
                )
            )
            Oval = self._objective.assess(kappas)
            objective_fct.append(Oval)
            phis.append(self._objective.rf.get_phi())
            self._ovals.append(Oval)

            if callback is not None:
                callback(counter)

            if log is not None:
                state = log.add_state(N_training=ActiveLearner.N_data)
                state.inform(self, active=ActiveLearner)
                state._kappa_samples_ref = kappas.detach().cpu()
                state["VI_params"] = {
                    k: v.cpu()
                    for k, v in self._objective._inference._q.state_dict().items()
                }
                state["objective"] = Oval

        return self._elbos, objective_fct, phis

    def run(self, ConvergenceCriteria, N_elbo_estimate=256, M_steps=1, callback=None):

        return self._run(
            ConvergenceCriteria, N_elbo_estimate, M_steps=M_steps, callback=callback
        )

    def _run(self, ConvergenceCriteria, N_elbo_estimate=256, M_steps=1, callback=None):

        self._objective.init()
        if self._counter == 0:
            self._log(init=True)

        t_init = time.time()

        abort = False
        n = 0
        l_elbos = list()

        while not abort:

            n += 1

            self._objective.E()

            J = self._objective.M(M_steps=M_steps)

            self._objective.step()

            self._log()

            self._counter += 1

            elbo = self._objective._inference.elbo_precise(N_elbo_estimate)
            l_elbos.append(elbo)
            abort = ConvergenceCriteria.step(elbo)

            if callback is not None:
                callback(n)

        self._elbos.append(l_elbos)

        return self._elbos


class IterationsConvergenceCriteria(object):
    def __init__(self, N):
        self._N = N
        self._counter = 0

    def reset(self):
        self._counter = 0

    def __bool__(self):
        return self._counter < self._N

    def step(self, *args, **kwargs):
        self._counter += 1
        abort = self._counter > self._N
        if abort:
            self.reset()
        return abort


class AdaptiveConvergenceCriteria(object):

    # based on ReduceLROnPlateau from PyTorch
    def __init__(
        self,
        patience=10,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=None,
        mode="max",
        num_max_steps=None,
        throw_error_if_max_steps_exceeded=False,
        verbose=False,
    ):

        assert threshold_mode in ["rel", "abs"]
        assert isinstance(threshold, float) and threshold > 0
        assert isinstance(patience, int) and patience > 0

        if verbose:
            print("Initalized with patience: {}".format(patience))

        self.patience = patience
        self.best = None
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.last_epoch = 0
        self.local_last_epoch = None
        self.num_bad_epochs = None
        self.cooldown = cooldown
        self.cooldown_counter = None
        self.verbose = verbose
        self._very_verbose = False

        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )

        self._num_max_steps = num_max_steps
        self._throw_error_if_max_steps_exceed = throw_error_if_max_steps_exceeded

        self._metric_memory = list()
        self.reset()

        self._abort_state = False

    def __bool__(self):

        return not self._abort_state

    def reset(self):

        self.best = self.mode_worse
        self.num_bad_epochs = 0
        self.cooldown_counter = self.cooldown
        self.local_last_epoch = 0
        self._metric_memory.append(list())

    def step(self, metric, prevent_reset=False):

        current = float(metric)
        epoch = self.last_epoch + 1
        self.last_epoch += 1
        self.local_last_epoch += 1
        self._metric_memory[-1].append(current)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0

            if self._very_verbose:
                print("Improvement achieved beyond threshold.")

        else:
            self.num_bad_epochs += 1

            if self._very_verbose:
                print("No improvement --- bad epochs: {}".format(self.num_bad_epochs))

        if self.in_cooldown:
            if self._very_verbose:
                print("Still in cooldown. Resetting")
            self.num_bad_epochs = 0
            self.cooldown_counter -= 1

        if self.num_bad_epochs > self.patience:
            abort = True
            if self.verbose:
                print(
                    "Adaptive convergence criteria has conerged after {} iterations".format(
                        self.local_last_epoch
                    )
                )
        else:
            abort = False

        if self.local_last_epoch >= self._num_max_steps and not abort:
            abort = True
            if self.verbose:
                print(
                    "Adaptive convergence criteria has exceeded maximum number of iterations ({})".format(
                        self._num_max_steps
                    )
                )
            if self._throw_error_if_max_steps_exceed:
                raise RuntimeError(
                    "No convergence could be achieved after the predetermined maximum number of steps ({})".format(
                        epochs
                    )
                )

        if abort:
            self.reset()
            self._abort_state = True
        else:
            self._abort_state = False

        return abort

    @property
    def in_cooldown(self):

        if self.cooldown is None:
            return False

        return self.cooldown_counter > 0

    def is_better(self, a, best):

        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):

        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        inf = float("inf")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
