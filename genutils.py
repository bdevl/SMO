import numpy as np
import glob
from scipy.stats import gaussian_kde
import logging
import logging.config
import os
import subprocess
import functools
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib.pyplot as plt
from datetime import date
import socket


# user and hostname where development work is being done
_USER_ = "user"
_HOSTNAME_WORKSTATION_ = "workstation"


__all__ = [
    "GridPlot",
    "ensure_folder",
    "date_folder",
    "pickle_save",
    "pickle_load",
    "Directory",
    "ETA",
    "EmptyShell",
    "backend",
]


def backend():

    """
    Returns True if code is run on a machine (e.g. cluster, GPU workstation) for which we do not want e.g. tqdm outputs
    """

    return socket.gethostname() != _HOSTNAME_WORKSTATION_


def cluster():

    """

    Returns True if code is run on the cluster

    """

    hostname = socket.gethostname()
    return "node" in hostname or "master" in hostname


class EmptyShell(object):
    def __init__(self, **kwargs):

        self.__dict__ = kwargs
        for kwarg, value in kwargs.items():
            self.__dict__[kwarg] = value

    def __repr__(self):

        s = "Container with the following members: \n"
        for key, value in self.__dict__.items():
            s += "{}  :  {} \n".format(key, value)
        return s


class ETA(object):
    def __init__(self, N, delay=False):

        self._t_init = None
        self._N = N
        self._n = None
        self._t = None

        if not delay:
            self.start()

    def start(self):
        self._t_init = time.time()

    def __call__(self, n):

        self._n = n
        self._t = time.time()

    def __repr__(self):

        raise NotImplementedError


class bcolors:

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def printd(msg):

    msg = "DEBUG: " + msg
    print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")


class Directory(object):

    # conveniency class
    def __init__(self, path=""):

        self._path = ensure_folder(path)

    @classmethod
    def phd(cls, path):

        basefolder = date_folder("")
        path = basefolder + path
        return cls(path)

    def savefig(self, subpath, *args, prevent_overwrite=False, **kwargs):

        if prevent_overwrite:
            raise NotImplementedError

        plt.savefig(self(subpath), *args, **kwargs)

    def __call__(self, subpath):

        path = self._path + subpath

        # if not a folder, reduce to folder
        folderpath = os.path.dirname(path)
        # and create, if it does not exist
        os.makedirs(folderpath, exist_ok=True)

        # if a directory (i.e. no filetype), append slash
        if len(path.split("/")[-1].split(".")) == 1:
            if path[-1] != "/":
                path += "/"

        return path

    def __repr__(self):
        return "Folder: " + self._path


def date_folder(foldername: str, basefolder=None) -> str:

    if basefolder is None:
        basefolder = "/home/" + _USER_ + "/phd/output/"
    folder = get_date_string()
    if foldername:
        fullpath = basefolder + folder + "/" + foldername + "/"
    else:
        fullpath = basefolder + folder + "/"

    fullpath = ensure_folder(fullpath)
    return fullpath


def get_date_string(str=None):

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")

    if str is not None:
        today_str += "-" + str

    return today_str


class CumulativeFloat(object):
    def __init__(self):

        self._entity = None
        self._counter = 0

    def reset(self):

        self._entity = None
        self._counter = 0

    def __float__(self):
        return self._entity / self._counter

    def __add__(self, other):

        assert isinstance(
            other, float
        ), "CumulativeFloat only works with floats (duh!)."

        if self._entity is None:
            self._entity = other
        else:
            self._entity += other

        self._counter += 1

        return self

    def __repr__(self):
        return str(self._entity / self._counter)


class MatplotlibStyle(object):
    def __init__(self, style=None):

        # bloated object doing little; implemented as object for future proofing
        if style is None:
            style = "seaborn"

        assert (
            style in plt.style.available
        ), "matplotlib style {} is not available".format(style)

        self._style = style

        self._setup()

    def _setup(self):
        plt.style.use(self._style)


def pickle_save(object, filename):

    """

    Args:
        object:     picke-able python object
        filename:   string, e.g. ~/subfolder/data.pickle

    Returns:

    """

    with open(filename, "wb") as handle:
        pickle.dump(object, handle)


def pickle_load(filename):

    """

    Args:
        filename: string, e.g. ~/subfolder/data.pickle

    Returns:

    """

    with open(filename, "rb") as handle:
        object = pickle.load(handle)

    return object


class GridPlot(object):

    # wrapper for grid plots
    def __init__(self, nrows, ncols, figsize=None):

        self.fig = plt.figure(constrained_layout=True, figsize=figsize)
        self.gs = self.fig.add_gridspec(nrows, ncols)
        self._nrows = nrows
        self._ncols = ncols

        # figure axes of subplots
        self._figaxi = list()

    @classmethod
    def FromRowSpec(cls, rowspec, figsize=None):

        ncols = np.lcm.reduce(rowspec)
        nrows = len(rowspec)

        for spec in rowspec:
            assert ncols % spec == 0

        num_columns_per_row_and_plot = [int(ncols / num_plots) for num_plots in rowspec]

        gridplot = cls(nrows, ncols, figsize=figsize)
        for rownumber, spec in enumerate(rowspec):
            for num_plot in range(spec):
                num_columns = num_columns_per_row_and_plot[rownumber]
                ax = gridplot.fig.add_subplot(
                    gridplot.gs[
                        rownumber, num_plot * num_columns : (num_plot + 1) * num_columns
                    ]
                )
                gridplot._figaxi.append(ax)

        return gridplot

    def add_subplot(self, *args, **kwargs):

        fxa = self.fig.add_subplot(*args, **kwargs)
        self._figaxi.append(fxa)

    def __getitem__(self, item):

        ax = self._figaxi[item]
        plt.axes(ax)
        return ax

    def __str__(self):
        return "A {} x {} grid plot".format(self._nrows, self._ncols)


def ensure_folder(path):

    path = path if path.endswith("/") else path + "/"
    os.makedirs(path, exist_ok=True)
    return path


class Study(object):
    def __init__(self, name=None):

        self._name = name
        self._ydata = dict()
        self._xdata = dict()

    def _check_identifier(self, identifier):

        if identifier not in self._xdata:
            self._xdata[identifier] = list()
            self._ydata[identifier] = list()

    def add(self, identifier, yval, xval=None):

        self._check_identifier(identifier)
        self._ydata[identifier].append(yval)
        if xval is not None:
            self._xdata[identifier].append(xval)

    def __getitem__(self, item):
        return self._ydata[item]

    def x(self, identifier):
        return self._xdata[identifier]

    def y(self, identifier):
        return self._ydata[identifier]


def kde(x_data, x_plot, bw_method=None, weights=None, plot=False):

    """

    Construct density estimate of x_data at x_plot points

    """

    kernel = gaussian_kde(x_data)
    pdf = kernel(x_plot)

    if plot:
        plt.plot(x_plot, pdf)

    return x_plot, pdf


class State(object):
    def __init__(self):
        pass


class StateVariable(object):
    def __init__(self, state):

        self._state = state

    def set(self, value):

        self._state = value


class ParameterInstance(object):
    def __init__(self, parameter, instance):

        self._parameter = parameter
        self._instance = instance


class Parameter(object):

    # see: https://github.com/elcorto/psweep
    _paramlist = list()

    def __init__(self, identifier, iterable, forgive=False):

        assert isinstance(identifier, str)

        if identifier in Parameter._paramlist:
            if forgive:
                return Parameter._paramlist[identifier]
            else:
                raise RuntimeError("Parameter {} already exists".format(identifier))

        self.identifier = identifier
        self._iterable = iterable

    def __iter__(self):
        for element in self._iterable:
            yield element

    def __eq__(self, other):
        if isinstance(other, str):
            return self.identifier == other
        elif isinstance(other, Parameter):
            return self.identifier == other.identifier
        else:
            raise RuntimeError(
                "Equality cannot be established for Parameter and object of type {}".format(
                    type(other)
                )
            )

    def __mul__(self, other):
        return ParameterProduct(self, other)

    def __add__(self, other):
        return ParameterUnion(self, other)

    def __repr__(self):
        return self.identifier


class BinaryParameter(Parameter):
    def __init__(self, identifier, forgive=False):
        super().__init__(identifier, [False, True], forgive=forgive)


class ConstantParameter(object):
    def __init__(self, identifier, value):
        super().__init__(identifier, [value], forgive=True)


class ParameterUnion(object):
    def __init__(self, p1, p2, repeat=False, interleave=False):

        # special case: ConstantParameter
        assert len(p1) == len(p2)
        self._p1 = None


class ParameterProduct(Parameter):
    def __init__(self, p1, p2):

        self._p1 = p1
        self._p2 = p2

    def __iter__(self):
        for p1 in self._p1:
            for p2 in self._p2:
                yield (p1, p2)

    def dictionary(self):

        combinations = list()
        for p1 in self._p1:
            for p2 in self._p2:
                params = dict()
                params[self._p1.identifier] = p1
                params[self._p2.identifier] = p2
                combinations.append(params)

        return combinations

    def __repr__(self):
        return "(" + str(self._p1) + " x " + str(self._p2) + ")"
