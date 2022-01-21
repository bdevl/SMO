import os
import time
import uuid
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from genutils import ensure_folder
from genutils import pickle_save, pickle_load, cluster

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 11,
        "text.latex.preamble": [r"\usepackage{bm}"],
    }
)


class AggregateConvergencePlotter(object):
    def __init__(
        self,
        basefolder: str,
        identifier: str,
        descriptors: List[str],
        construct_iterator=True,
        path=None,
    ):

        if path is not None:
            path = ensure_folder(path)

        # meta-information
        self._basefolder = basefolder
        self._identifier = identifier
        self._descriptors = descriptors

        # setup
        self.objective = dict()
        self.dfactor = dict()
        self.N_training = dict()
        self.phis = dict()
        self.elbos = dict()

        self._setup()

    def _setup(self):

        for descriptor in self._descriptors:
            (
                n_training,
                obj,
                phi_trajectory,
                elbos,
                dfactor_,
            ) = AggregateConvergencePlotter.extract(
                self._basefolder, self._identifier, descriptor
            )
            self.objective[descriptor] = obj
            self.N_training[descriptor] = n_training
            self.phis[descriptor] = phi_trajectory
            self.elbos[descriptor] = elbos
            self.dfactor[descriptor] = dfactor_

    def _extract_qoi(self, container):

        if not all([len(elem) == len(container[0]) for elem in container]):
            lengths_data_entries = np.array([len(elem) for elem in container])
            most_common = np.argmax(np.bincount(lengths_data_entries))
            mfraction = np.mean((lengths_data_entries == most_common).astype(np.float))
            warnings.warn(
                "Only using results with {} entries (corresponds to fraction of data : {:.2f}".format(
                    most_common, mfraction
                )
            )
            qoi = [elem for elem in container if len(elem) == most_common]
        else:
            qoi = [elem for elem in container]

        assert qoi, "no entries were recovered"

        # qoi will be a list of lists
        return qoi

    def plot(
        self,
        reduce=np.mean,
        title=None,
        xlabel=None,
        ylabel=None,
        labels=None,
        fig=None,
        legendloc=None,
    ):

        if fig is None:
            fig = plt.figure()

        assert labels is None or (
            isinstance(labels, list) and len(labels) == len(self._descriptors)
        )

        xlabel = "N : labeled data (microstructures)" if xlabel is None else xlabel
        ylabel = r"achieved metric $\mathcal{S}$" if ylabel is None else ylabel
        title = "Convergence - Metric" if title is None else title

        objective_f = dict()
        k = 0
        for j, descriptor in enumerate(self._descriptors):

            # data_entries = [len(elem) for elem in self.objective[descriptor]]
            objective_f[descriptor] = self._extract_qoi(self.objective[descriptor])
            objective_ = reduce(np.array(objective_f[descriptor]), 0)

            assert len(objective_) != len(
                objective_f[descriptor]
            ), "unequal length. something has gone wrong."

            if labels is None:
                mlabel = descriptor
            else:
                mlabel = labels[j]

            plt.plot(self.N_training[descriptor], objective_, "o-", label=mlabel)
            plt.grid(True)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend(loc=legendloc)

    @staticmethod
    def extract(basefolder, identifier, descriptor, construct_iterator=True):

        N_cases = Log.N_cases(basefolder, identifier, descriptor)
        if N_cases < 1:
            raise RuntimeError(
                "Could not load any logs for: {} : {} : {}".format(
                    basefolder, identifier, descriptor
                )
            )

        if construct_iterator:

            def logs():
                return Log.LoadAllCasesIterator(basefolder, identifier, descriptor)

        else:
            logs_list = Log.LoadAllCases(basefolder, identifier, descriptor)

            def logs():
                return logs_list

        objective = list()
        phi = list()
        elbos = list()
        dfactor = list()

        for n, log in enumerate(logs()):

            objective.append(list())
            phi.append(list())
            elbos.append(list())
            dfactor.append(list())

            # dirty fix
            if n == 0:
                if log.states[0].N_training is not None:
                    N_training = [state.N_training for state in log.states]
                else:
                    N_training = np.arange(len(log.states))

            for state in log.states:
                objective[-1].append(state["objective"])
                phi[-1].append(state["phi"])
                elbos[-1].append(state["elbos"])
                try:
                    dfactor[-1].append(state["dfactor"])
                except:
                    pass

        return N_training, objective, phi, elbos, dfactor


class PhiTrajectory(object):
    def __init__(self, states: None):

        if states is not None:
            assert isinstance(states, list), "expects list of states"
            assert all(
                [isinstance(state, PhiState) for state in states]
            ), "expects list of > PhiStates < "

        self._states = list() if states is None else states

    @classmethod
    def FromNumpyArray(
        cls, phi_trajectory_np: np.ndarray, skip=None, iterations=None, wmodel=None
    ):

        assert isinstance(phi_trajectory_np, np.ndarray) and phi_trajectory_np.ndim == 2

        if skip is not None:
            trajectory = cls(
                [
                    PhiState(phi_trajectory_np[m, :], wmodel=wmodel)
                    for m in range(phi_trajectory_np.shape[0], skip)
                ]
            )
        elif iterations is not None:
            assert isinstance(iterations, list)
            phi_trajectory_np = phi_trajectory_np[iterations, :]
            trajectory = cls(
                [
                    PhiState(phi_trajectory_np[m, :], wmodel=wmodel)
                    for m in range(phi_trajectory_np.shape[0])
                ]
            )
            assert len(trajectory) == len(iterations)
        else:
            trajectory = cls(
                [
                    PhiState(phi_trajectory_np[m, :], wmodel=wmodel)
                    for m in range(phi_trajectory_np.shape[0])
                ]
            )

        return trajectory

    def precompute_model(self, N: int):

        assert self._states, "trajectory does not have any states"

        for state in tqdm(self._states):
            state.demand_kappas_model(N)

    def precompute_ref(self, N: int):

        assert self._states, "trajectory does not have any states"

        for state in tqdm(self._states):
            state.demand_kappas_ref(N)

    def __getitem__(self, item):
        return self._states[item]

    def __len__(self):
        return len(self._states)

    def sdf(self, wmodel, path, movie=False, skip=None):

        phi_orig = deepcopy(wmodel.rf.get_phi())

        figs = list()
        for n, state in enumerate(self._states):

            if skip is None or n % skip == 0:
                fig = plt.figure()
                wmodel.rf.set_phi(state.phi)
                wmodel.rf.kernel.PlotFrequencyDomain(
                    title=" | Iteration : {}".format(n)
                )

                figs.append(fig)

        wmodel.rf.set_phi(phi_orig)

    def min(self, mode) -> np.ndarray:

        assert mode in ("model", "reference")

        for n, state in enumerate(self._states):

            kappa_min = state.min(mode)

            if n == 0:
                kappa_min_global = kappa_min
            else:
                kappa_min_global = np.minimum(kappa_min_global, kappa_min)

        return kappa_min_global

    def max(self, mode) -> np.ndarray:

        assert mode in ("model", "reference")

        for n, state in enumerate(self._states):

            kappa_max = state.max(mode)

            if n == 0:
                kappa_max_global = kappa_max
            else:
                kappa_max_global = np.maximum(kappa_max_global, kappa_max)

        return kappa_max_global

    def append(self, state):
        assert isinstance(state, PhiState)
        self._states.append(state)


class PhiState(object):
    def __init__(
        self, phi: np.ndarray, kappas_ref=None, kappas_model=None, wmodel=None
    ):

        self._phi = phi
        self.kappas_ref = kappas_ref
        self.kappas_model = kappas_model
        self._wmodel = wmodel

    @property
    def wmodel(self):

        if self._wmodel is None:
            raise RuntimeError(
                "demand_kappas_ref() requires wmodel (with valid hmg set)"
            )

        return self._wmodel

    def prune(self):
        self._wmodel = None

    @property
    def phi(self):
        return self._phi

    def plot(self):

        assert self._wmodel is not None

        phi_orig = deepcopy(self._wmodel.rf.get_phi())
        self._wmodel.rf.set_phi(self._phi)
        self._wmodel.rf.kernel.PlotFrequencyDomain()
        self._wmodel.rf.set_phi(phi_orig)

    def min(self, mode):

        return self._bound(mode, np.min)

    def max(self, mode):

        return self._bound(mode, np.max)

    def _bound(self, mode, minmax):

        if mode == "model":
            assert self.kappas_model is not None
            return minmax(self.kappas_model, 0)
        elif mode == "reference":
            assert self.kappas_ref is not None
            return minmax(self.kappas_ref, 0)
        else:
            raise RuntimeError

    def demand_kappas_ref(self, N):

        if self.kappas_ref is not None and N <= self.kappas_ref.shape[0]:
            return self.kappas_ref[0:N]
        elif self.kappas_ref is not None and N > self.kappas_ref.shape[0]:
            N_add = N - self.kappas_ref.shape[0]
            kappa_add_samples = (
                self.wmodel.sample_reference_distribution(N=N_add, phi=self.phi)
                .detach()
                .cpu()
                .numpy()
            )
            self.kappas_ref = np.concatenate((self.kappas_ref, kappa_add_samples), 0)
            assert self.kappas_ref.ndim == 2 and self.kappas_ref.shape[0] == N
            return self.kappas_ref
        elif self.kappas_ref is None:
            self.kappas_ref = (
                self.wmodel.sample_reference_distribution(N=N, phi=self.phi)
                .detach()
                .cpu()
                .numpy()
            )
            return self.kappas_ref

    def demand_kappas_model(self, N):

        if self.kappas_model is not None and N <= self.kappas_model.shape[0]:
            return self.kappas_model[0:N]
        elif self.kappas_model is not None and N > self.kappas_model.shape[0]:
            N_add = N - self.kappas_model.shape[0]
            kappa_add_samples = (
                self.wmodel.sample_predicted_distribution(N=N_add, phi=self.phi)
                .detach()
                .cpu()
                .numpy()
            )
            self.kappas_model = np.concatenate(
                (self.kappas_model, kappa_add_samples), 0
            )
            assert self.kappas_model.ndim == 2 and self.kappas_model.shape[0] == N
            return self.kappas_model
        elif self.kappas_model is None:
            self.kappas_model = (
                self.wmodel.sample_predicted_distribution(N=N, phi=self.phi)
                .detach()
                .cpu()
                .numpy()
            )
            return self.kappas_model

    @property
    def kappas_ref(self):
        return self._kappas_ref

    @kappas_ref.setter
    def kappas_ref(self, value):
        assert (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            or isinstance(value, type(None))
        )
        self._kappas_ref = value

    @property
    def kappas_model(self):
        return self._kappas_model

    @kappas_model.setter
    def kappas_model(self, value):
        assert (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            or isinstance(value, type(None))
        )
        self._kappas_model = value


class Log(object):
    def __init__(self, identifier: str, descriptor: str, uuid_: str = None):

        self._identifier = identifier
        self._descriptor = descriptor
        self._states = list()
        self._filepath = None
        self.data = dict()

        if uuid_ is None:
            uuid_ = str(uuid.uuid4())

        self._uuid = uuid_
        self._t1 = time.time()
        self._t2 = None

    @property
    def uuid(self):
        return self._uuid

    def touch(self, toggle=None):

        if toggle is None:
            toggle = cluster()

        if toggle:
            Path(os.getcwd() + "/" + self.uuid).touch()

    def _get_runtime(self):
        if self._t1 is None or self._t2 is None:
            raise RuntimeError("Cannot provide the runtime")

        timestr = time.strftime("%H:%M:%S", time.gmtime(self._t2 - self._t1))
        return timestr

    @property
    def runtime(self):
        return self._get_runtime()

    def finalize(self):

        self._t2 = time.time()

    @property
    def identifier(self):
        return self._identifier

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def states(self):
        return self._states

    def __getitem__(self, item):
        return self._states[item]

    @staticmethod
    def MatchCases(directory: str, identifier: str, descriptor: str) -> list:

        assert isinstance(identifier, str) and isinstance(descriptor, str)
        filenamestartswith = identifier + "_" + descriptor

        identified_files = list()

        for file in os.listdir(directory):
            if file.startswith(filenamestartswith):
                identified_files.append(file)

        return identified_files

    @classmethod
    def LoadAllCases(cls, directory, identifier, descriptor, return_only_first=False):

        files = cls.MatchCases(directory, identifier, descriptor)

        logs = list()
        for n in range(len(files)):
            log = cls.FromFile(directory + files[n])
            if return_only_first:
                return log
            logs.append(log)

        return logs

    @classmethod
    def N_cases(cls, directory, identifier, descriptor):

        files = cls.MatchCases(directory, identifier, descriptor)
        return len(files)

    @classmethod
    def LoadCaseNumber(cls, directory, identifier, descriptor, n):

        files = cls.MatchCases(directory, identifier, descriptor)
        assert n < len(files), "There are only {} log files".format(len(files))

        log = cls.FromFile(directory + files[n])
        return log

    @classmethod
    def LoadUUID(cls, directory, uuid_):

        assert isinstance(uuid_, str)
        try:
            uuid_obj = uuid.UUID(uuid_, version=4)
            assert str(uuid_obj) == uuid_
        except Exception:
            raise ValueError("The uuid string is not a valid format.")

        desired_file = None
        for file in os.listdir(directory):
            if uuid_ in file:
                assert (
                    desired_file is None
                ), "There is more than one file with the specified UUID"
                desired_file = file

        if desired_file is None:
            raise ValueError("The requested UUID could not be found.")

        return cls.FromFile(directory + desired_file)

    @classmethod
    def LoadAllCasesIterator(cls, directory, identifier, descriptor, filter=None):

        files = cls.MatchCases(directory, identifier, descriptor)
        for n in range(len(files)):

            if filter is not None:
                log = cls.FromFile(directory + files[n])
                if filter(log):
                    yield log
            else:
                yield cls.FromFile(directory + files[n])

    @classmethod
    def FromFile(cls, path):

        sequence = pickle_load(path)
        sequence._filepath = path
        return sequence

    def add_state(self, *args, **kwargs):

        state = State(*args, **kwargs)
        self.append(state)
        return state

    def append(self, state):

        self._states.append(state)
        state._uuid = self._uuid

    def save(self, directory):

        if self._t2 is None:
            self.finalize()

        directory = ensure_folder(directory)
        filename = self._identifier + "_" + self._descriptor + "_" + self._uuid
        fullpath = directory + filename

        pickle_save(self, fullpath)


class State(object):
    def __init__(self, N_training: int, uuid_: str = None):
        assert isinstance(uuid_, str) or uuid_ is None
        self._N_training = N_training
        self._data = dict()
        self._kappa_samples_ref = None
        self._kappa_samples_surr = None
        self._uuid = uuid_

    @property
    def N_training(self):
        return self._N_training

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def inform(self, em, active=None):
        objective = em.objective
        wmodel = objective.wmodel

        self["modelstate"] = deepcopy(wmodel.state())
        self["elbos"] = deepcopy(em._elbos)
        self["phi"] = self._phi_trajectory = deepcopy(em.phi_trajectory)

        if active is not None:
            self["dfactor"] = deepcopy(active.dfactor)
