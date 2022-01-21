import fenics as df


class UnitMeshHelper(object):
    def __init__(self, mesh, V=None, Vc=None, pDegree=None, isotropic=None):

        assert mesh.geometric_dimension() in [
            1,
            2,
        ], "Can only deal with 1-D or 2-D meshes"
        self._mesh = mesh

        if V is None:
            assert pDegree is not None and pDegree in [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ], "polynomial degree does not comply"
            self._V = df.FunctionSpace(mesh, "CG", pDegree)
        else:
            assert pDegree is None, "cannot specify pDegree if V has been passed"
            self._V = V

        if Vc is None:
            assert (
                isotropic is not None
            ), "If Vc is not given, need to specify whether isotropic or not"
            if isotropic:
                self._Vc = df.FunctionSpace(mesh, "DG", 0)
            else:
                self._Vc = df.TensorFunctionSpace(mesh, "DG", 0, symmetry=True)
        else:
            assert (
                isotropic is None
            ), "cannot specify isotropic or anisotropic behaviour if Vc is passed explicitly"
            self._Vc = Vc

        self.boundaries = None
        self.dx = None
        self.ds = None
        self._has_been_setup = False

    def __call__(self, comm=None):

        assert not self._has_been_setup, "Cannot call the UnitMeshHelper a second time."

        if comm is None:
            comm = self._mesh.mpi_comm()

        if self._mesh.geometric_dimension() == 1:
            return self._setup_1D(comm)
        elif self._mesh.geometric_dimension() == 2:
            return self._setup_2D(comm)
        else:
            raise RuntimeError

        self._has_been_setup = True

    def _setup_2D(self, comm):

        boundaries = dict()
        boundaries["left"] = df.CompiledSubDomain(
            "near(x[0], 0.0) && on_boundary", mpi_comm=comm
        )
        boundaries["bottom"] = df.CompiledSubDomain(
            "near(x[1], 0.0) && on_boundary", mpi_comm=comm
        )
        boundaries["top"] = df.CompiledSubDomain(
            "near(x[1], 1.0) && on_boundary", mpi_comm=comm
        )
        boundaries["right"] = df.CompiledSubDomain(
            "near(x[0], 1.0) && on_boundary", mpi_comm=comm
        )

        boundarymarkers = df.MeshFunction(
            "size_t", self._mesh, self._mesh.topology().dim() - 1, 0
        )
        boundarymarkers.set_all(0)
        domainmarkers = df.MeshFunction(
            "size_t", self._mesh, self._mesh.topology().dim(), 0
        )

        boundaries["left"].mark(boundarymarkers, 1)
        boundaries["bottom"].mark(boundarymarkers, 2)
        boundaries["right"].mark(boundarymarkers, 3)
        boundaries["top"].mark(boundarymarkers, 4)

        ds = df.Measure("ds", domain=self._mesh, subdomain_data=boundarymarkers)
        dx = df.Measure("dx", domain=self._mesh, subdomain_data=domainmarkers)

        self.dx = dx
        self.ds = ds

        return boundaries, boundarymarkers, domainmarkers, dx, ds, self._V, self._Vc

    def _setup_1D(self, comm):

        boundaries = dict()
        boundaries["left"] = df.CompiledSubDomain(
            "near(x[0], 0.0) && on_boundary", mpi_comm=comm
        )
        boundaries["right"] = df.CompiledSubDomain(
            "near(x[0], 1.0) && on_boundary", mpi_comm=comm
        )

        boundarymarkers = df.MeshFunction(
            "size_t", self._mesh, self._mesh.topology().dim() - 1, 0
        )
        boundarymarkers.set_all(0)
        domainmarkers = df.MeshFunction(
            "size_t", self._mesh, self._mesh.topology().dim(), 0
        )

        boundaries["left"].mark(boundarymarkers, 1)
        boundaries["right"].mark(boundarymarkers, 2)

        ds = df.Measure("ds", domain=self._mesh, subdomain_data=boundarymarkers)
        dx = df.Measure("dx", domain=self._mesh, subdomain_data=domainmarkers)

        return boundaries, boundarymarkers, domainmarkers, dx, ds, self._V, self._Vc
