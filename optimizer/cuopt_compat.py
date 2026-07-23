"""Version-tolerant shim over the cuOpt Python API.

cuOpt's Python layer was reorganized between the 25.12, 26.02 and 26.04/26.06
releases (LP/QP/MILP was split into "Convex Optimization" and "MIP" sections,
and enum locations moved). Rather than scatter try/except across the optimizer,
every version-sensitive lookup is isolated here and asserted once at import.

API surface targeted (verified against the 26.02 reference):

    from cuopt.linear_programming.problem import (
        Problem, Variable, LinearExpression, QuadraticExpression,
        Constraint, VType, CType, sense,
    )
    from cuopt.linear_programming.solver_settings import SolverSettings

    prob = Problem("name")
    x = prob.addVariable(lb=0.0, ub=1.0, obj=0.0, vtype=VType.CONTINUOUS, name="x")
    prob.addConstraint(x + y >= 1)
    prob.setObjective(QuadraticExpression(matrix, prob.getVariables()), sense=MINIMIZE)
    prob.solve(settings)
    prob.Status, prob.ObjValue, prob.SolveTime, x.getValue()
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass


class CuOptUnavailable(RuntimeError):
    """Raised when cuOpt is not importable, with a message that says what to do."""


@dataclass(frozen=True)
class CuOptApi:
    Problem: type
    QuadraticExpression: type
    LinearExpression: type
    Constraint: type
    SolverSettings: type
    VType: object
    CType: object
    MINIMIZE: object
    MAXIMIZE: object
    version: str


@functools.lru_cache(maxsize=1)
def load_cuopt() -> CuOptApi:
    """Import cuOpt and normalize the names this project uses.

    Raises ``CuOptUnavailable`` with an actionable message on any CPU-only
    machine, which is the expected state during development.
    """
    try:
        problem_mod = importlib.import_module("cuopt.linear_programming.problem")
        settings_mod = importlib.import_module("cuopt.linear_programming.solver_settings")
    except ImportError as exc:  # pragma: no cover - exercised only off-GPU
        raise CuOptUnavailable(
            "cuOpt is not installed in this environment. This module only runs on a "
            "CUDA machine. Install with:\n"
            "  pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu13==26.2.*'\n"
            "and pin the version you verified against. On a CPU-only host, use "
            "optimizer.mean_variance_cpu instead."
        ) from exc

    def pick(module, *names):
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        raise CuOptUnavailable(
            f"none of {names} found in {module.__name__}; the cuOpt API has changed. "
            f"Re-check docs.nvidia.com/cuopt and update optimizer/cuopt_compat.py."
        )

    # `sense` is an enum in some releases and MINIMIZE/MAXIMIZE are re-exported
    # at module level in others; accept either.
    if hasattr(problem_mod, "MINIMIZE"):
        minimize, maximize = problem_mod.MINIMIZE, problem_mod.MAXIMIZE
    else:
        sense = pick(problem_mod, "sense", "Sense", "ObjSense")
        minimize, maximize = sense.MINIMIZE, sense.MAXIMIZE

    try:
        version = importlib.import_module("cuopt").__version__
    except Exception:
        version = "unknown"

    return CuOptApi(
        Problem=pick(problem_mod, "Problem"),
        QuadraticExpression=pick(problem_mod, "QuadraticExpression"),
        LinearExpression=pick(problem_mod, "LinearExpression"),
        Constraint=pick(problem_mod, "Constraint"),
        SolverSettings=pick(settings_mod, "SolverSettings"),
        VType=pick(problem_mod, "VType", "VariableType"),
        CType=pick(problem_mod, "CType", "ConstraintType"),
        MINIMIZE=minimize,
        MAXIMIZE=maximize,
        version=version,
    )


def cuopt_available() -> bool:
    try:
        load_cuopt()
        return True
    except CuOptUnavailable:
        return False


def status_name(prob) -> str:
    """cuOpt reports Status as an int in current releases, an enum in others.

    (The commonly-circulated `prob.Status.name` snippet raises AttributeError
    on 26.02, where Status is a plain int.)
    """
    status = prob.Status
    if hasattr(status, "name"):
        return str(status.name)
    return {1: "Optimal", 2: "Infeasible", 3: "Unbounded"}.get(int(status), f"Status({status})")


def is_optimal(prob) -> bool:
    name = status_name(prob).lower()
    return "optimal" in name


@functools.lru_cache(maxsize=1)
def quadratic_convention() -> float:
    """Determine whether cuOpt's quadratic matrix means x'Qx or (1/2)x'Qx.

    Solvers disagree on this by a factor of two (CVXPY's ``quad_form`` is x'Qx;
    several QP solvers use the 1/2 convention), and getting it wrong silently
    halves or doubles the effective risk aversion — the portfolio still looks
    plausible, so the bug survives eyeballing. Rather than assume, solve a
    one-variable problem with a known answer under each convention:

        minimize  q*x^2 - c*x   with q = c = 1, x in [0, 10]
        argmin = 0.5  if the matrix means x'Qx
        argmin = 1.0  if it means (1/2) x'Qx

    Returns the multiplier to apply to the covariance matrix so the realized
    objective is x'Sigma x.
    """
    api = load_cuopt()
    prob = api.Problem("convention-probe")
    x = prob.addVariable(lb=0.0, ub=10.0)
    prob.setObjective(api.QuadraticExpression([[1.0]], [x]) + (-1.0) * x, sense=api.MINIMIZE)
    prob.solve(api.SolverSettings())

    value = float(x.getValue())
    if abs(value - 0.5) < 1e-3:
        return 1.0  # matrix already means x'Qx
    if abs(value - 1.0) < 1e-3:
        return 2.0  # matrix means (1/2)x'Qx -> double it
    raise CuOptUnavailable(
        f"quadratic-convention probe returned x={value}, expected 0.5 or 1.0. "
        f"Do not trust objective comparisons until this is resolved."
    )
