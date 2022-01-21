import numpy as np
from dolfin import DirichletBC
import dolfin as df


class BoundaryEncodingEnsemble(object):
    def __init__(self, boundary_encodings):
        self._boundary_encodings = boundary_encodings

    def __getitem__(self, item):
        return self._boundary_encodings[item]

    def __iter__(self):
        yield from self._boundary_encodings


class BoundaryEncoding(object):
    def __init__(self, dirichlet_encoding, neumann_encoding):

        assert isinstance(dirichlet_encoding, DirichletBoundaryEncoding)
        assert isinstance(neumann_encoding, NeumannBoundaryEncoding)

        self.dirichlet_encoding = dirichlet_encoding
        self.neumann_encoding = neumann_encoding

    def reconstruct(self):
        raise NotImplementedError


class DirichletBoundaryEncoding(object):
    def __init__(self, type, data=None):
        self.type = type

        if data is None:
            self._data = dict()
        else:
            assert isinstance(data, dict)
            self._data = data

    @property
    def data(self):
        return self._data

    def __getitem__(self, item):
        try:
            return self._data[item]
        except KeyError:
            raise KeyError(
                "DirichletBoundaryEncoding could not retrieve '{}'".format(item)
            )

    def __setitem__(self, key, value):
        self._data[key] = value

    def reconstruct(self, factory):
        # returns Dirichlet Boundary encoding
        return factory.reconstruct_dirichlet(self)


class NeumannBoundaryEncoding(object):
    def __init__(self, type, data=None):
        self.type = type

        if data is None:
            self._data = dict()
        else:
            assert isinstance(data, dict)
            self._data = data

    @property
    def data(self):
        return self._data

    def __getitem__(self, item):
        try:
            return self._data[item]
        except KeyError:
            raise KeyError(
                "NeumannBoundaryEncoding could not retrieve '{}'".format(item)
            )

    def __setitem__(self, key, value):
        self._data[key] = value

    def reconstruct(self, factory):

        return factory.reconstruct_neumann(self)


class DirichletSpecification(object):
    def __init__(self, expression, domain, component=None, pointwise=False):

        self.expression = expression
        self.domain = domain
        self.component = component
        self.pointwise = pointwise

    def is_homogeneous(self):
        raise NotImplementedError

    def mark_facetfct_with_value(self, facetfct, value):

        assert value >= 0 and isinstance(value, int)
        self.domain.mark(facetfct, value)


class DirichletBoundaryCondition(object):
    def __init__(self, bcs, encoding=None, encoding_type=None, encoding_data=None):

        if isinstance(bcs, list):
            for bc in bcs:
                if not isinstance(bc, DirichletSpecification):
                    raise TypeError(
                        "DirichletBoundaryCondition expects (list of) Dirichlet Specifications"
                    )
        elif isinstance(bcs, DirichletSpecification):
            bcs = [bcs]
        else:
            raise TypeError(
                "DirichletBoundaryCondition expects (list of) Dirichlet Specifications"
            )

        self._bcs = bcs  # list of DirichletSpecifications

        if encoding is not None:
            assert encoding_type is None
            assert encoding_data is None

        if encoding is not None:
            assert isinstance(encoding, DirichletBoundaryEncoding)
            self._encoding = encoding

        if encoding_data is not None or encoding_type is not None:
            assert encoding_data is not None and encoding_type is not None
            assert isinstance(encoding_type, str)
            assert isinstance(encoding_data, dict)
            self._encoding = DirichletBoundaryEncoding(encoding_type, encoding_data)

    def __getitem__(self, ind):
        return self._bcs[ind]

    def encode(self):
        if self._encoding is None:
            raise NotImplementedError(
                "No encodings of the DirichletBoundaryCondition have been set in the factory. Cannot encode."
            )
        return self._encoding

    def extract(self, V, ReturnFreeDofs=False):

        fbcs = self.transfer(V)
        dofs = np.array(
            [dof for bc in fbcs for dof in bc.get_boundary_values().keys()], dtype=int
        )
        vals = np.array(
            [val for bc in fbcs for val in bc.get_boundary_values().values()],
            dtype=float,
        )

        dofs, index = np.unique(dofs, return_index=True)
        values = vals[index]

        if ReturnFreeDofs:
            all_dofs = set(V.dofmap().dofs())
            free_dofs = np.array(list(all_dofs - set(dofs)), dtype=np.int)
            return dofs, values, free_dofs

        return dofs, values

    def is_homogeneous(self, V):

        dofs, values = self.extract(V)
        return not any(values)

    def transfer(self, V):

        fenics_bcs = list()

        for bc in self._bcs:
            if bc.component is not None:
                if not bc.pointwise:
                    fenics_bcs.append(
                        DirichletBC(V.sub(bc.component), bc.expression, bc.domain)
                    )
                else:
                    fenics_bcs.append(
                        DirichletBC(
                            V.sub(bc.component),
                            bc.expression,
                            bc.domain,
                            method="pointwise",
                        )
                    )
            else:
                if not bc.pointwise:
                    fenics_bcs.append(DirichletBC(V, bc.expression, bc.domain))
                else:
                    fenics_bcs.append(
                        DirichletBC(V, bc.expression, bc.domain, method="pointwise")
                    )

        return fenics_bcs

    def mark_facets(self, mesh):

        facetfct = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        facetfct.set_all(0)
        for bc in self._bcs:
            bc.domain.mark(facetfct, 1)

        return facetfct

    def mark_facetfct_with_value(self, facetfct, value):

        assert value >= 0 and isinstance(value, int)
        for bc in self._bcs:
            bc.domain.mark(facetfct, value)

    def apply(self, X):
        raise NotImplementedError


class NeumannSpecification(object):
    def __init__(self, type, expression, subdomain=None):

        if type not in ["ds", "dx"]:
            raise ValueError('Type must either be "ds" or "dx')

        self._type = type  # e.g. ds
        self._subdomain = subdomain
        self._expression = expression

    @property
    def type(self):
        return self._type

    @property
    def subdomain(self):
        return self._subdomain

    @property
    def expression(self):
        return self._expression

    def mark_facetfct_with_value(self, facetfct, value):

        assert value >= 0 and isinstance(value, int)
        if self._type == "ds":
            self.subdomain.mark(facetfct, value)
        else:
            raise RuntimeError(
                "cannot mark facetfcts for dx type Neumann Specification"
            )


class NeumannBoundaryCondition(object):
    def __init__(
        self,
        NeumannSpecifications,
        encoding=None,
        encoding_type=None,
        encoding_data=None,
    ):

        self._neumman_specifications = NeumannSpecifications

        if encoding is not None:
            assert isinstance(encoding, NeumannBoundaryEncoding)
            self._encoding = encoding

        if encoding_data is not None or encoding_type is not None:
            assert encoding_data is not None and encoding_type is not None
            assert isinstance(encoding_type, str)
            assert isinstance(encoding_data, dict)
            self._encoding = NeumannBoundaryEncoding(encoding_type, encoding_data)

    def encode(self):
        if self._encoding is None:
            raise NotImplementedError(
                "No encodings of the NeumannBoundaryCondition have been set in the factory. Cannot encode."
            )
        return self._encoding

    def __getitem__(self, ind):
        return self._neumman_specifications[ind]

    def compile_form(self, V):

        mesh = V.mesh()
        v = df.TestFunction(V)
        form = None

        meshfct = dict()
        meshfct["ds"] = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        meshfct["dx"] = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        meshfct["ds"].set_all(0)
        meshfct["dx"].set_all(0)

        ID = dict()
        ID["ds"] = 1
        ID["dx"] = 1

        for ns in self._neumman_specifications:

            if ns.subdomain is None:
                # entire domain, no restrictions
                ID_ = 0
            else:
                ID_ = ID[ns.type]
                ns.subdomain.mark(meshfct[ns.type], ID_)
                ID[ns.type] += 1
                assert ID_ != ID[ns.type]

            measure = df.Measure(
                ns.type, domain=mesh, subdomain_data=meshfct[ns.type], subdomain_id=ID_
            )

            form_ = ns.expression * v * measure

            if form is None:
                form = form_
            else:
                form = form + form_

        return form

    def assemble_flux(self, V):

        return df.assemble(self.compile_form(V)).get_local()

    def mark_facetfct_with_value(self, facetfct, value):

        assert value >= 0 and isinstance(value, int)
        for nsp in self.self._neumman_specifications:
            nsp.domain.mark(facetfct, value)
