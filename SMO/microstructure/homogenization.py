import numpy as np
import dolfin as df

df.set_log_level(50)
import torch
from ufl import nabla_div
import matplotlib.pyplot as plt
from SMO.microstructure.fawkes import UnitMeshHelper
from fawkes.converter import DiscontinuousGalerkinPixelConverter
from typing import Union, List


def optimize_loadcases(loadcases: list, target: List[set], greedy: bool = True) -> list:

    # solve set cover problem
    assert isinstance(loadcases, list)
    assert all([isinstance(loadcase, set) for loadcase in loadcases])
    assert isinstance(target, set)

    assert (
        set().union(*loadcases) == target
    ), "Loadcases are not able to satisfy desired targets"

    if greedy:
        covered = set()
        covering = list()
        while covered != target:
            loadcase = max(loadcases, key=lambda s: len(s - covered))
            covering.append(loadcase)
            covered |= loadcase

        return [loadcases.index(covering_) for covering_ in covering]

    else:
        raise NotImplementedError


class PoissonHomogenizer(object):
    def __init__(
        self, Vc, V=None, htype=("xx", "xy", "yy"), pDegree=2, integration_type="volume"
    ):

        assert integration_type in ["boundary", "volume"]
        assert pDegree in [1, 2, 3, 4, 5, 6]
        assert (
            Vc.mesh().geometric_dimension() == 2
        ), "PoissonHomogenizer currently can only deal with 2D meshes"
        assert (
            Vc.dolfin_element().signature()
            == "FiniteElement('Discontinuous Lagrange', triangle, 0)"
        )

        helper = UnitMeshHelper(Vc.mesh(), Vc=Vc, V=V, pDegree=pDegree)
        boundaries, boundarymarkers, domainmarkers, dx, ds, V, Vc = helper()

        admissible = set(("xx", "xy", "yy"))
        self._admissible = tuple(admissible)
        assert isinstance(htype, str) or isinstance(htype, tuple)
        assert htype in admissible or set(htype).issubset(admissible)
        self._htype = htype

        self._V = V
        self._Vc = Vc
        self._mesh = Vc.mesh()
        self._dx = dx
        self._ds = ds
        self._alpha = df.Function(self._Vc)
        self._boundaries = boundaries
        self._a = None
        self._L = None
        self._pDegree = pDegree
        self._integration_type = integration_type
        self._pixelconverter = None
        self._setup()

    @property
    def admissible(self):
        return self._admissible

    @classmethod
    def FromImageResolution(cls, N, *args, **kwargs):

        mesh = df.UnitSquareMesh(df.MPI.comm_self, N, N)
        Vc = df.FunctionSpace(mesh, "DG", 0)
        hmg = cls(Vc, *args, **kwargs)
        converter = DiscontinuousGalerkinPixelConverter(Vc)
        converter.assemble()
        hmg._pixelconverter = converter
        return hmg

    def _setup(self):

        u = df.TrialFunction(self._V)
        v = df.TestFunction(self._V)
        self._a = df.inner(self._alpha * df.grad(u), df.grad(v)) * self._dx
        self._L = df.Constant(0.0) * v * self._dx

    @property
    def mesh(self):
        return self._mesh

    @property
    def gdim(self):
        return self._mesh.geometric_dimension()

    def _set_x(self, x):

        assert x.ndim == 1 and len(x) == self._Vc.dim()
        assert np.all(x > 0)
        self._alpha.vector()[:] = x

    @classmethod
    def FromPhysics(cls, physics):
        return cls(physics.Vc)

    def homogenize_img(self, X, *, AcknowledgeRaw=False) -> Union[dict, List[dict]]:

        if self._pixelconverter is None:
            raise RuntimeError("Pixel converter has not been set")

        if X.dim() < 3:
            X.unsqueeze_(0)

        if not isinstance(X, torch.Tensor):
            assert isinstance(X, np.ndarray)
            X = torch.tensor(X)
        assert (X.dim() == 3 and X.shape[1] == X.shape[2]) or (
            X.dim() == 2 and X.shape[0] == X.shape[1]
        )
        with torch.no_grad():
            X = self._pixelconverter.ImageToFunctionBatchedFast(X)

        if X.shape[0] == 1:
            X.squeeze_()

        return self._homogenize(X.detach().cpu().numpy(), AcknowledgeRaw=AcknowledgeRaw)

    def __call__(self, X_raw, *, AcknowledgeRaw=False):

        return self._homogenize(X_raw, AcknowledgeRaw=AcknowledgeRaw)

    def _homogenize(self, X_raw, *, AcknowledgeRaw=False) -> Union[dict, List[dict]]:

        if not AcknowledgeRaw:
            raise RuntimeError("Please acknowledge that you are passing raw X values")

        assert isinstance(X_raw, np.ndarray)
        if X_raw.ndim == 1:
            return self._solve_load_cases(X_raw)
        else:
            assert X_raw.shape[1] == self._Vc.dim() and X_raw.shape[0] > 1
            kappas = list()
            for x_raw in X_raw:
                # consider changing to yield statement
                kappas.append(self._solve_load_cases(x_raw))

        return kappas

    def _solve_load_cases(self, x: np.ndarray) -> dict:

        self._set_x(x)
        kappa = dict()

        if self._integration_type == "volume":

            comm = self._mesh.mpi_comm()

            u_boundary = df.Expression("x[0]", degree=1, mpi_comm=comm)
            on_boundary = df.CompiledSubDomain("on_boundary", mpi_comm=comm)
            bc = df.DirichletBC(self._V, u_boundary, on_boundary)
            u0 = df.Function(self._V)
            df.solve(self._a == self._L, u0, bc)

            q1 = self._alpha * df.grad(u0)

            form1 = q1[0] * df.dx
            form2 = q1[1] * df.dx

            kappa["xx"] = df.assemble(form1)
            kappa["xy"] = df.assemble(form2)

            u_boundary = df.Expression("x[1]", degree=1, mpi_comm=comm)
            on_boundary = df.CompiledSubDomain("on_boundary", mpi_comm=comm)
            bc = df.DirichletBC(self._V, u_boundary, on_boundary)
            u0 = df.Function(self._V)
            df.solve(self._a == self._L, u0, bc)

            q1 = self._alpha * df.grad(u0)
            form3 = q1[1] * df.dx
            kappa["yy"] = df.assemble(form3)

            return kappa

        else:
            raise RuntimeError(
                'Integration type "{}" unknown or deprecated.'.format(
                    self._integration_type
                )
            )


class CombinedHomogenizer(object):
    def __init__(self, homogenizers, properties: Union[None, List[dict]] = None):

        assert isinstance(homogenizers, list)
        assert len(homogenizers) == 2, "we currently only support two homogenizers"
        self._homogenizers = homogenizers

        if properties is not None:
            assert len(properties) == len(homogenizers)
            for property in properties:
                assert (
                    isinstance(property, dict)
                    and "high" in property
                    and "low" in property
                )
        self._properties = properties

        assert all([hasattr(homogenizer, "admissible") for homogenizer in homogenizers])

    @property
    def admissible(self) -> tuple:

        R = set()
        for homogenizer in self._homogenizers:
            r = set(homogenizer.admissible)
            assert not R.intersection(
                r
            ), "there is a overlap in the admissible type of base homogenizers. we do not want this (for now)"
            R = R.union(r)

        return tuple(R)

    def homogenize_img(
        self, X: torch.Tensor, *, AcknowledgeRaw=False
    ) -> Union[dict, List[dict]]:

        results = list()

        if self._properties is not None:
            # inputs have to be a 1/0 mask of phases
            assert torch.all(
                X.unique(sorted=True)
                == torch.tensor([0.0, 1.0], dtype=X.dtype, device=X.device)
            ), "the homogenizer needs to be provided with a 1/0 mask of high / low phase properties"

        for i, hom in enumerate(self._homogenizers):
            # returns either a dictionary, or list of dictionaries
            if self._properties is None:
                results.append(hom.homogenize_img(X, AcknowledgeRaw=AcknowledgeRaw))
            else:
                # deal with different properties for each homogenizer
                Xp = torch.zeros_like(X)
                Xp[X == 1.0] = self._properties[i]["high"]
                Xp[X == 0.0] = self._properties[i]["low"]
                results.append(hom.homogenize_img(Xp, AcknowledgeRaw=AcknowledgeRaw))

        if isinstance(results[0], dict):
            assert (
                len(results) == 2
            ), "current implementation only supports two underlying homogenizers"
            return {**results[0], **results[1]}
        else:
            assert (
                len(results) == 2
            ), "current implementation only supports two underlying homogenizers"
            # return list of fused dictionaries
            return [{**results[0][n], **results[1][n]} for n in range(len(results[0]))]


def ConvertToLameParameters(E, nu):

    mu = 0.5 * (1 / (1 + nu)) * E
    lmbda = (nu / (1 - 2 * nu)) * (1 / (1 + nu)) * E
    return mu, lmbda


class LinearHomogenizer(object):
    def __init__(self, mesh, htype=None, pDegree=1, integration_type="volume"):

        self._mesh = mesh
        self._pDegree = pDegree
        self._integration_type = integration_type

        admissible = set(
            ["C{}{}".format(i, j) for i in range(3) for j in range(3)]
            + ["mu", "lambda", "E", "nu", "E_avg"]
        )
        self._admissible = tuple(admissible)
        if htype is None:
            htype = self._admissible
        else:
            assert isinstance(htype, str) or isinstance(htype, tuple)
            assert htype in admissible or set(htype).issubset(admissible)

        self._htype = htype
        self._Vc = df.FunctionSpace(mesh, "DG", 0)
        self._mu = df.Function(self._Vc)
        self._lambda = df.Function(self._Vc)
        self._u_sol = None
        self._C_hom = np.zeros((3, 3))
        self._mu_hom = None
        self._lamda_hom = None
        self._E_hom = None
        self._nu_hom = None
        self._E_avg = None
        self._volume = df.assemble(df.Constant(1) * df.Measure("dx", domain=self._mesh))
        self._area = df.assemble(df.Constant(1) * df.Measure("ds", domain=self._mesh))
        self._load_case = None
        self._pixelconverter = None
        self._nu_default_value = None

    @property
    def admissible(self):
        return self._admissible

    @classmethod
    def FromImageResolution(cls, nx, *args, **kwargs):

        mesh = df.UnitSquareMesh(df.MPI.comm_self, nx, nx)
        hmg = cls(mesh, *args, **kwargs)
        converter = DiscontinuousGalerkinPixelConverter(hmg._Vc)
        converter.assemble()
        hmg._pixelconverter = converter
        return hmg

    @property
    def Vc(self):
        return self._Vc

    @property
    def C_hom(self):
        return self._C_hom

    @property
    def gdim(self):

        return self._mesh.geometric_dimension()

    def set_parameters(self, E, nu=None, high_phase=None, low_phase=None):

        if nu is None:
            assert (
                self._nu_default_value is not None
            ), "There is no default value set for nu (and only E has been passed)"

        if isinstance(E, float):
            Et = E
        else:
            assert high_phase is not None and low_phase is not None

            e_min = E.min()
            e_max = E.max()

            if e_min != e_max:
                assert e_min == 0 and e_max == 1
            else:
                assert e_min == 0 or e_min == 1

            T1 = np.abs(E - 1) < 1e-12
            T2 = np.abs(E) < 1e-12
            Et = np.zeros(E.shape)
            Et[T1] = high_phase
            Et[T2] = low_phase

        mu, lmbda = ConvertToLameParameters(Et, nu)
        self._set_fct_from_object(self._mu, mu)
        self._set_fct_from_object(self._lambda, lmbda)

    def set_young_modulus_from_vector(self, E):

        assert (
            self._nu_default_value is not None
        ), "There is no default value set for nu (and only E has been passed)"
        assert isinstance(E, np.ndarray)
        mu, lmbda = ConvertToLameParameters(E, self._nu_default_value)
        self._set_fct_from_object(self._mu, mu)
        self._set_fct_from_object(self._lambda, lmbda)

    def _set_fct_from_object(self, fct, obj):

        if isinstance(obj, float):
            fct.vector()[:] = obj
        elif isinstance(obj, np.ndarray) and obj.ndim == 1:
            assert (
                len(obj) == fct.function_space().dim()
            ), "vector does not match function space dimension"
            fct.vector()[:] = obj.copy()
        elif isinstance(obj, np.ndarray) and obj.ndim == 2:
            assert obj.size == int(
                0.5 * fct.function_space().dim()
            ), "2D image and DG function space do not match"
            raise NotImplementedError
        else:
            raise TypeError("unknown type for data object to be set to function space")

    def _loadcase_macro_strain(self, n):

        assert 0 <= n <= 2, "there are only 3 elementary load cases, n=0,1,2"
        ev = np.zeros(3)
        ev[n] = 1
        Epsilon_ = ((ev[0], 0.5 * ev[2]), (0.5 * ev[2], ev[1]))
        return df.Constant(Epsilon_)

    def _sigma_homogenized(self, boundary=False):
        def my_cross(v, w):
            return df.as_vector((v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2]))

        if boundary:
            raise DeprecationWarning
        else:
            sigma_xx = df.assemble(self._sigma(self._u_sol)[0, 0] * df.dx)
            sigma_yy = df.assemble(self._sigma(self._u_sol)[1, 1] * df.dx)
            sigma_xy = df.assemble(self._sigma(self._u_sol)[1, 0] * df.dx)

        return np.array([sigma_xx, sigma_yy, sigma_xy])

    def __call__(self, *args, **kwargs):
        self.homogenize(*args, **kwargs)

    def _homogenize(self, verbose=False, plot=False, boundary=False):

        for n in range(3):

            self.solve_load_case(n)
            self._C_hom[n, :] = self._sigma_homogenized(boundary=boundary)

            if plot:
                self.plot()

        self._mu_hom = self._C_hom[2, 2]
        self._lamda_hom = self._C_hom[0, 1]

        self._E_hom = (
            self._mu_hom
            * (3 * self._lamda_hom + 2 * self._mu_hom)
            / (self._lamda_hom + self._mu_hom)
        )

        self._nu_hom = self._lamda_hom / (self._lamda_hom + self._mu_hom) / 2

        self._E_avg = 0.5 * (self._C_hom[0, 0] + self._C_hom[1, 1])

        if verbose:
            print(self)

    def _collect_results_in_dict(self):

        r = dict()

        for i in range(3):
            for j in range(3):
                r["C{}{}".format(i, j)] = self._C_hom[i, j]

        r["mu"] = self._mu_hom
        r["lambda"] = self._lamda_hom
        r["E"] = self._E_hom
        r["nu"] = self._nu_hom
        r["E_avg"] = self._E_avg

        assert set(r.keys()) == set(
            self.admissible
        ), "returned homogenized properties does not match the admissible set."

        return r

    def homogenize(self, X, *, AcknowledgeRaw=False):

        if not AcknowledgeRaw:
            raise RuntimeError("Acknowledge that you are passing raw X values")

        if self._integration_type == "boundary":
            boundary = True
        elif self._integration_type == "volume":
            boundary = False
        else:
            raise ValueError(
                'Integration type {} not recognized (either "boundary" or "volume"'.format(
                    integration_type
                )
            )

        assert isinstance(X, np.ndarray)
        if X.ndim == 1:

            self.set_young_modulus_from_vector(X)
            self._homogenize(verbose=False, plot=False, boundary=boundary)
            kappa = self._collect_results_in_dict()
            return kappa
        else:
            assert X.shape[1] == self._Vc.dim() and X.shape[0] > 1
            kappas = list()
            for x_raw in X:
                self.set_young_modulus_from_vector(x_raw)
                self._homogenize(verbose=False, plot=False, boundary=boundary)
                kappa_ = self._collect_results_in_dict()
                kappas.append(kappa_)

            return kappas

    def homogenize_img(self, X, *, AcknowledgeRaw=False) -> Union[dict, List[dict]]:

        if self._pixelconverter is None:
            raise RuntimeError("Pixel converter has not been set")

        assert isinstance(X, torch.Tensor), "homogenize_img expects a torch tensor"

        if X.dim() < 3:
            X.unsqueeze_(0)

        if not isinstance(X, torch.Tensor):
            assert isinstance(X, np.ndarray)
            X = torch.tensor(X)

        assert (X.dim() == 3 and X.shape[1] == X.shape[2]) or (
            X.dim() == 2 and X.shape[0] == X.shape[1]
        )
        with torch.no_grad():
            X = self._pixelconverter.ImageToFunctionBatchedFast(X)

        if X.shape[0] == 1:
            X.squeeze_()

        return self.homogenize(X.detach().cpu().numpy(), AcknowledgeRaw=AcknowledgeRaw)

    def solve_load_case(self, n):

        loadcases = ["Exx", "Eyy", "Exy"]
        self._load_case = loadcases[n]
        Epsilon = self._loadcase_macro_strain(n)

        self._solve(Epsilon)

    def __repr__(self):

        if self._E_hom is not None:

            s = " =========================================== \n"
            s += "Effective Poissons ratio: {} \n".format(self._nu_hom)

            if self._E_hom < 1e3:
                unit = "Pa"
                E_hom = self._E_hom
            elif self._E_hom < 1e6:
                unit = "KPa"
                E_hom = self._E_hom / 1e3
            elif self._E_hom < 1e9:
                unit = "MPa"
                E_hom = self._E_hom / 1e6
            else:
                unit = "GPa"
                E_hom = self._E_hom / 1e9

            s += "Effective Bulk modulus: {:.4f} {} \n".format(E_hom, unit)
            s += "Tangential moduli: \n"
            s += np.array_str(self._C_hom, precision=2)

            s += " \n"
            s += " =========================================== \n"

        else:

            s = "Homogenizer (has not yet been called)"

        return s


class PeriodicHomogenizer(LinearHomogenizer):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        a = 1
        b = 1
        c = 0

        vertices = np.array([[0, 0.0], [a, 0.0], [a + c, b], [c, b]])

        self._Ve = df.VectorElement("CG", self._mesh.ufl_cell(), self._pDegree)
        self._Re = df.VectorElement("R", self._mesh.ufl_cell(), 0)
        self._W = df.FunctionSpace(
            self._mesh,
            df.MixedElement([self._Ve, self._Re]),
            constrained_domain=PeriodicBoundary(vertices, tolerance=1e-10),
        )
        self._V = df.FunctionSpace(self._mesh, self._Ve)
        self._v_, self._lamb_ = df.TestFunctions(self._W)
        self._dv, self._dlamb = df.TrialFunctions(self._W)
        self._w = df.Function(self._W)
        self._Epsilon = df.Constant(((0, 0), (0, 0)))
        self._F = (
            df.inner(self._sigma_fluctations(self._dv), self._epsilon(self._v_)) * df.dx
        )
        self._a, self._L = df.lhs(self._F), df.rhs(self._F)
        self._a += (
            df.dot(self._lamb_, self._dv) * df.dx
            + df.dot(self._dlamb, self._v_) * df.dx
        )
        self._v_sol = None

    def _epsilon(self, u):
        return df.sym(df.grad(u))

    def _sigma_fluctations(self, v):

        return self._lambda * df.tr(self._epsilon(v) + self._Epsilon) * df.Identity(
            self.gdim
        ) + 2 * self._mu * (self._epsilon(v) + self._Epsilon)

    def _sigma_homogenized(self, boundary=True):

        if boundary:
            raise DeprecationWarning
        else:

            sigma_xx = (1 / self._volume) * df.assemble(
                self._sigma_fluctations(self._v_sol)[0, 0] * df.dx
            )
            sigma_yy = (1 / self._volume) * df.assemble(
                self._sigma_fluctations(self._v_sol)[1, 1] * df.dx
            )
            sigma_xy = (1 / self._volume) * df.assemble(
                self._sigma_fluctations(self._v_sol)[1, 0] * df.dx
            )

        return np.array([sigma_xx, sigma_yy, sigma_xy])

    def _solve(self, Epsilon):

        self._Epsilon.assign(Epsilon)

        df.solve(
            self._a == self._L,
            self._w,
            [],
            solver_parameters={"linear_solver": "petsc"},
        )

        (self._v_sol, lamb) = df.split(self._w)

        y = df.SpatialCoordinate(self._mesh)
        self._u_sol = 0.5 * (self._v_sol + df.dot(self._Epsilon, y))


class PeriodicBoundary(df.SubDomain):
    def __init__(self, vertices, tolerance=df.DOLFIN_EPS):

        # NOTE: External code taken over / adopted from: https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html

        df.SubDomain.__init__(self, tolerance)

        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1, :] - self.vv[0, :]
        self.a2 = self.vv[3, :] - self.vv[0, :]
        assert np.linalg.norm(self.vv[2, :] - self.vv[3, :] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2, :] - self.vv[1, :] - self.a2) <= self.tol

    def inside(self, x, on_boundary):

        return bool(
            (
                df.near(
                    x[0], self.vv[0, 0] + x[1] * self.a2[0] / self.vv[3, 1], self.tol
                )
                or df.near(
                    x[1], self.vv[0, 1] + x[0] * self.a1[1] / self.vv[1, 0], self.tol
                )
            )
            and (
                not (
                    (
                        df.near(x[0], self.vv[1, 0], self.tol)
                        and df.near(x[1], self.vv[1, 1], self.tol)
                    )
                    or (
                        df.near(x[0], self.vv[3, 0], self.tol)
                        and df.near(x[1], self.vv[3, 1], self.tol)
                    )
                )
            )
            and on_boundary
        )

    def map(self, x, y):

        if df.near(x[0], self.vv[2, 0], self.tol) and df.near(
            x[1], self.vv[2, 1], self.tol
        ):
            y[0] = x[0] - (self.a1[0] + self.a2[0])
            y[1] = x[1] - (self.a1[1] + self.a2[1])
        elif df.near(x[0], self.vv[1, 0] + x[1] * self.a2[0] / self.vv[2, 1], self.tol):
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]


def macro_strain(i):

    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array(
        [[Eps_Voigt[0], Eps_Voigt[2] / 2.0], [Eps_Voigt[2] / 2.0, Eps_Voigt[1]]]
    )


def stress2Voigt(s):

    return df.as_vector([s[0, 0], s[1, 1], s[0, 1]])
