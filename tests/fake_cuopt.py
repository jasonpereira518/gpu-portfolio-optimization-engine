"""Stand-in for the slice of cuOpt's Python modeling API this project uses.

It reproduces the model-building semantics the optimizer depends on, as they
appear in NVIDIA/cuopt v26.08.00
(python/cuopt/cuopt/linear_programming/problem.py):

* ``Problem.setObjective`` first zeroes every variable's linear objective
  coefficient — including any passed as ``addVariable(obj=...)`` — and then
  adds the expression's own linear terms.
* A matrix-form ``QuadraticExpression`` is added *positionally* onto a
  NumVariables x NumVariables matrix, so it has to cover every variable.
* ``QuadraticExpression + LinearExpression`` concatenates coefficient lists
  with ``+`` (an ndarray there would be added elementwise instead).

The objective the solver would receive is exposed as ``linear_objective`` (c)
and ``quadratic_objective`` (Q, dense). Nothing is solved: the GPU parity
tests remain the ground truth for what cuOpt returns.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from optimizer.cuopt_compat import CuOptApi

MINIMIZE, MAXIMIZE = "minimize", "maximize"


class Constraint:
    def __init__(self, expr: "LinearExpression", sense: str, rhs: float) -> None:
        self.expr, self.sense, self.rhs = expr, sense, float(rhs)


class LinearExpression:
    def __init__(self, vars, coefficients, constant) -> None:
        self.vars = vars
        self.coefficients = coefficients
        self.constant = constant

    def __add__(self, other):
        if isinstance(other, LinearExpression):
            return LinearExpression(self.vars + other.vars,
                                    self.coefficients + other.coefficients,
                                    self.constant + other.constant)
        if isinstance(other, Variable):
            return LinearExpression(self.vars + [other], self.coefficients + [1.0], self.constant)
        return LinearExpression(self.vars, self.coefficients, self.constant + float(other))

    __radd__ = __add__

    def __neg__(self):
        return LinearExpression(self.vars, [-c for c in self.coefficients], -self.constant)

    def __sub__(self, other):
        return self + (-other)

    def __ge__(self, rhs):
        return Constraint(self, ">=", rhs)

    def __le__(self, rhs):
        return Constraint(self, "<=", rhs)

    def __eq__(self, rhs):  # noqa: D105 — builds a constraint, like cuOpt
        return Constraint(self, "==", rhs)

    __hash__ = None


class Variable:
    def __init__(self, index: int, lb: float, ub: float, obj: float, vtype, name: str) -> None:
        self._index = index
        self.lb, self.ub, self.vtype, self.name = lb, ub, vtype, name
        self._obj = float(obj)

    def getIndex(self) -> int:
        return self._index

    def getObjectiveCoefficient(self) -> float:
        return self._obj

    def setObjectiveCoefficient(self, value: float) -> None:
        self._obj = float(value)

    def _expr(self) -> LinearExpression:
        return LinearExpression([self], [1.0], 0.0)

    def __add__(self, other):
        return self._expr() + other

    __radd__ = __add__

    def __sub__(self, other):
        return self._expr() - other

    def __neg__(self):
        return LinearExpression([self], [-1.0], 0.0)

    def __mul__(self, k):
        return LinearExpression([self], [float(k)], 0.0)

    __rmul__ = __mul__


class QuadraticExpression:
    def __init__(self, qmatrix=None, qvars=[], qvars1=[], qvars2=[], qcoefficients=[],  # noqa: B006
                 vars=[], coefficients=[], constant=0.0) -> None:  # noqa: B006 — mirrors cuOpt
        self.qmatrix = None
        self.qvars = qvars
        if qmatrix is not None:
            self.qmatrix = coo_matrix(qmatrix)
            if self.qmatrix.shape[0] != self.qmatrix.shape[1]:
                raise ValueError("qmatrix should be a square matrix")
            if len(qvars) != self.qmatrix.shape[0]:
                raise ValueError("qvars length mismatch. Should match with qmatrix length.")
        self.qvars1, self.qvars2, self.qcoefficients = qvars1, qvars2, qcoefficients
        self.vars, self.coefficients, self.constant = vars, coefficients, constant

    def __add__(self, other):
        if isinstance(other, LinearExpression):
            vars_, coeffs = self.vars + other.vars, self.coefficients + other.coefficients
            constant = self.constant + other.constant
        elif isinstance(other, Variable):
            vars_, coeffs, constant = self.vars + [other], self.coefficients + [1.0], self.constant
        else:
            vars_, coeffs = self.vars, self.coefficients
            constant = self.constant + float(other)
        return QuadraticExpression(self.qmatrix, self.qvars, self.qvars1, self.qvars2,
                                   self.qcoefficients, vars_, coeffs, constant)

    __radd__ = __add__


class Problem:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.vars: list[Variable] = []
        self.constraints: list[tuple[str, Constraint]] = []
        self.sense = None
        self.quadratic_objective: np.ndarray | None = None

    @property
    def NumVariables(self) -> int:
        return len(self.vars)

    def addVariable(self, lb=0.0, ub=float("inf"), obj=0.0, vtype="CONTINUOUS", name=""):
        var = Variable(len(self.vars), lb, ub, obj, vtype, name)
        self.vars.append(var)
        return var

    def getVariables(self) -> list[Variable]:
        return list(self.vars)

    def addConstraint(self, constraint: Constraint, name: str = "") -> Constraint:
        self.constraints.append((name, constraint))
        return constraint

    def setObjective(self, expr, sense=MINIMIZE) -> None:
        self.sense = sense
        for var in self.vars:
            var.setObjectiveCoefficient(0.0)

        if isinstance(expr, Variable):
            expr = expr._expr()
        for var, coeff in zip(expr.vars, expr.coefficients):
            target = self.vars[var.getIndex()]
            target.setObjectiveCoefficient(target.getObjectiveCoefficient() + coeff)

        n = self.NumVariables
        q = coo_matrix((n, n))
        if isinstance(expr, QuadraticExpression):
            rows = [v.getIndex() for v in expr.qvars1]
            cols = [v.getIndex() for v in expr.qvars2]
            q = coo_matrix((np.array(expr.qcoefficients, dtype=float), (rows, cols)), shape=(n, n))
            if expr.qmatrix is not None:
                q = q + expr.qmatrix  # scipy raises ValueError on a shape mismatch
        self.quadratic_objective = q.toarray()

    @property
    def linear_objective(self) -> np.ndarray:
        return np.array([v.getObjectiveCoefficient() for v in self.vars])


FAKE_API = CuOptApi(
    Problem=Problem,
    QuadraticExpression=QuadraticExpression,
    LinearExpression=LinearExpression,
    Constraint=Constraint,
    SolverSettings=object,
    VType=None,
    CType=None,
    MINIMIZE=MINIMIZE,
    MAXIMIZE=MAXIMIZE,
    version="fake",
)
