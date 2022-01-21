import dolfin as df
import math


class ScalarExpressionFromFunction(df.UserExpression):
    def __init__(self, f, **kwargs):

        self._f = f
        super().__init__(**kwargs)

    def eval(self, values, x):

        f_val = self._f(x)
        values[0] = f_val

    def value_shape(self):
        return ()


class CustomSubdomain(df.SubDomain):
    def __init__(self, f):

        self._expr = ScalarExpressionFromFunction(f)

    def inside(self, x, boundary):
        return self._expr(x)


class RadialBasisFunction(df.UserExpression):
    def __init__(self, r0, l, **kwargs):
        self.r0 = r0
        self.l = l
        super().__init__(**kwargs)

    def eval_cell(self, values, x, ufc_cell):
        raise NotImplementedError

    def eval(self, values, x):

        T = (x[0] - self.r0[0]) ** 2 + (x[1] - self.r0[1]) ** 2
        values[0] = math.exp((-T / self.l ** 2))

    def value_shape(self):
        return ()


def FastRadialBasisFunction(element):

    r0 = df.Constant((0.5, 0.5))
    l = df.Constant(0.15)
    return (
        df.Expression(
            " exp(-(pow((x[0] - r0[0]),2) + pow((x[1] - r0[1]),2))/ pow(l,2))",
            r0=r0,
            l=l,
            element=element,
        ),
        r0,
        l,
    )
