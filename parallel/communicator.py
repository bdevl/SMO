from mpi4py import MPI
import numpy as np
import time
from mpi4py.futures import MPIPoolExecutor
import torch


class Connector(object):
    def __init__(self, comm=MPI.COMM_WORLD):

        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()


class Pool(object):
    def __init__(self, connector):

        self._connector = connector

    def submit(self, f, args):
        pass


def GatherVerticalStack(A, comm, target=0, ReturnOrder="f"):

    size = comm.Get_size()
    rank = comm.Get_rank()

    sendbuf = A.flatten(order="f")
    recvbuf = None

    if rank == target:
        recvbuf = np.empty(shape=A.size * size, dtype=A.dtype)

    comm.Gather(sendbuf, recvbuf, root=target)

    if rank == target:
        if ReturnOrder == "f":  # column-wise ordering
            return recvbuf.reshape(A.shape[0], A.shape[1] * size, order="f")
        elif ReturnOrder == "c":  # row-wise ordering
            # clumsy, but...
            return np.ascontiguousarray(
                recvbuf.reshape(A.shape[0], A.shape[1] * size, order="f")
            )
        else:
            raise Exception(
                "No recognized array-order; expect either c (rows) or f (columns)"
            )
    else:
        # return None
        return recvbuf
