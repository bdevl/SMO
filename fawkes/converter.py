import fenics as df
import numpy as np
import torch


class DiscontinuousGalerkinPixelConverter(object):
    def __init__(self, V):

        self.V = V
        self.mesh = V.mesh()

        self._X = None
        self._Y = None

        self._dx = None
        self._dy = None

        self._Nx = None
        self._Ny = None

        #
        self.Interpolator = None
        self.ReverseInterpolator = None

    @property
    def px(self):

        if self._Nx is None:
            self._assemble()

        return self._Nx - 1

    @property
    def py(self):

        if self._Ny is None:
            self._assemble()

        return self._Ny - 1

    def assemble(self):
        self._assemble()

    def _assemble(self):

        if self.mesh.geometric_dimension() == 1:
            self._assemble_1D()
        elif self.mesh.geometric_dimension() == 2:
            self._assemble_2D()
        else:
            raise NotImplementedError("Cannot do 3D")

    def _assemble_1D(self):

        nodes = np.sort(self.mesh.coordinates().flatten())

        assert (
            nodes.max() == 1 and nodes.min() == 0
        ), "Implementation assumes unit inverall mesh [0,1]"

        diff = np.diff(nodes)
        self._dx = diff[0]
        assert np.all(np.abs(diff - self._dx) < 1e-12), "Mesh is not uniform"

        dofmap = self.V.dofmap()

        x_centers = np.zeros(self.V.dim())

        Interpolator = np.zeros(self.V.dim(), dtype=np.int32)

        for i, cell in enumerate(df.cells(self.mesh)):

            x = cell.midpoint().x()
            x_centers[i] = x

            pixel_id = int(x // self._dx)

            assert (
                dofmap.cell_dofs(i) == i
            ), "permutation of dg0 function space has to be taken into consideration"

            Interpolator[pixel_id] = i

        InterpolatorReverse = np.zeros(self.V.dim(), dtype=np.int32)

        for n in range(len(InterpolatorReverse)):
            loc = np.where(Interpolator == n)[0]
            InterpolatorReverse[n] = loc

        self._DofToPixelPermutator = torch.tensor(Interpolator, dtype=torch.long)
        self._PixelToDofPermutator = torch.tensor(InterpolatorReverse, dtype=torch.long)

    def _assemble_2D(self):

        coordinates = np.array(
            np.zeros(self.mesh.num_vertices()), dtype=[("x", float), ("y", float)]
        )
        for i, vertex in enumerate(df.vertices(self.mesh)):
            coordinates["x"][i] = vertex.x(0)
            coordinates["y"][i] = vertex.x(1)

        self._Ny = len(np.unique(coordinates["y"]))
        self._Nx = len(np.unique(coordinates["x"]))
        assert self._Ny * self._Nx == self.mesh.num_vertices()
        assert self._Ny == self._Nx

        coordinates = np.sort(coordinates, order=["y", "x"])

        X = coordinates["x"].reshape(self._Ny, self._Nx)
        Y = coordinates["y"].reshape(self._Ny, self._Nx)

        T = np.diff(X, axis=1)
        self._dx = T[0, 0]
        assert np.all(np.abs(T - self._dx) < 1e-12)
        T = np.diff(Y, axis=0)
        self._dy = T[0, 0]
        assert np.all(np.abs(T - self._dy) < 1e-12)

        self._X = np.flipud(X) + 0.5 * self._dx
        self._Y = np.flipud(Y) + 0.5 * self._dy

        Interpolator = np.zeros(((self._Ny - 1) * (self._Nx - 1), self.V.dim()))
        dofmap = self.V.dofmap()

        for i, cell in enumerate(df.cells(self.mesh)):

            x = cell.midpoint().x()
            y = cell.midpoint().y()

            assert dofmap.cell_dofs(i) == i

            cx = int(x // self._dx)
            cy = int(y // self._dy)

            cy = (self._Ny - 2) - cy
            pixel_id = cy * (self._Ny - 1) + cx

            Interpolator[pixel_id, i] = 0.5

        ReverseInterpolator = np.zeros((self.V.dim(), (self._Ny - 1) * (self._Nx - 1)))
        for i, row in enumerate(Interpolator):
            ind = np.where(row)[0]
            ReverseInterpolator[ind[0], i] = 1
            ReverseInterpolator[ind[1], i] = 1

        self.Interpolator = Interpolator
        self.ReverseInterpolator = ReverseInterpolator

        #
        a, b = np.where(self.Interpolator != 0)
        self._DofToPixelPermutator = torch.tensor(b, dtype=torch.long)

        a, b = self.ReverseInterpolator.nonzero()
        self._PixelToDofPermutator = torch.tensor(b, dtype=torch.long)

    def FunctionToImage(self, x, reshape=False):

        y = self.Interpolator @ x
        if reshape:
            y = y.reshape((self._Ny - 1, self._Nx - 1))
        return y

    def FunctionToImageBatchedFast(self, X):

        if X.dim() == 1:
            # deal with batch_size 1
            X = X.view(1, -1)

        batch_size = X.shape[0]
        X_perm = X[:, self._DofToPixelPermutator]
        Images_flattened = 0.5 * (X_perm[:, 0::2] + X_perm[:, 1::2])
        Images = Images_flattened.view(batch_size, self.py, self.px)

        return Images

    def ImageToFunctionBatchedFast(self, Images):

        batch_size = Images.shape[0]
        flattened_Images = Images.view(batch_size, -1)
        X = flattened_Images[:, self._PixelToDofPermutator]
        return X

    def ImageToFunction(self, y):

        if y.ndim == 2:
            y = y.flatten(order="C")  # default order: C

        return self.ReverseInterpolator @ y
