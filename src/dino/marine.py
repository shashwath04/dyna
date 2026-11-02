"""Adapters for nonlinear ODEs and Fossen-form marine equations.

These adapters assemble/solve supplied equations; they do not identify vessel
coefficients, generate hydrodynamic matrices, or assume a propulsion layout.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._validation import array, covariance, vector
from .types import Array


class MassMatrixTerms(Protocol):
    """Return M (n,n), rhs (n,) in M(t,x,u) x_dot = rhs(t,x,u).

    M must be nonsingular. Algebraic constraints/DAEs require a separate solver.
    Callbacks may capture parameters, environments, and nonlinear actuator laws.
    """

    def __call__(self, t: float, x: Array, u: Array) -> tuple[Array, Array]: ...


@dataclass(frozen=True)
class MassMatrixDynamics:
    """Expose a supplied mass-matrix ODE through the ordinary Dynamics interface."""

    terms: MassMatrixTerms

    def __call__(self, t: float, x: Array, u: Array) -> Array:
        if not np.isfinite(t):
            raise ValueError("time must be finite")
        x, u = vector(x, "x"), vector(u, "u")
        mass, rhs = self.terms(t, x.copy(), u.copy())
        mass = array(mass, "mass", (x.size, x.size))
        rhs = array(rhs, "rhs", x.shape)
        try:
            return array(np.linalg.solve(mass, rhs), "state derivative", x.shape)
        except np.linalg.LinAlgError as exc:
            raise ValueError("mass matrix must be nonsingular") from exc


@dataclass(frozen=True)
class FossenTerms:
    """Terms in eta_dot=k, M nu_dot=tau-C nu-D nu-g.

    M/C/D (d,d), g/tau (d,), kinematics (pose_size,). Units are SI:
    generalized forces N/N m; mixed translational/rotational inertia and rates.
    M includes rigid-body and added mass and must be symmetric positive definite.
    Matrices, forces and kinematics may depend on time, pose, velocity and input.
    """

    inertia: Array
    coriolis: Array
    damping: Array
    restoring: Array
    force: Array
    kinematics: Array


class FossenModel(Protocol):
    """Supply vessel/environment terms for pose eta, body velocity nu, command u.

    Callback arguments are independent copies. Compute kinematics explicitly, e.g.
    J(eta)@nu; quaternion poses can have 7 entries while nu has 6.
    """

    def __call__(self, t: float, eta: Array, nu: Array, u: Array) -> FossenTerms: ...


@dataclass(frozen=True)
class FossenDynamics:
    """Dynamics adapter for x=[eta,nu], with configurable pose/velocity sizes.

    Defaults to a planar 3-DOF vessel: eta=[north,east,yaw], nu=[surge,sway,yaw_rate].
    For Euler 6-DOF use (6,6); for quaternion poses use (7,6) and an appropriate
    custom stepper to preserve unit norm. Ocean-current-relative formulations
    must supply their complete forces/terms consistently; current effects are
    not automatically inferred. Augmented actuator states can use a custom
    Dynamics callback or MassMatrixDynamics instead.
    """

    model: FossenModel
    pose_size: int = 3
    velocity_size: int = 3

    def __post_init__(self) -> None:
        if any(type(n) is not int or n <= 0 for n in (self.pose_size, self.velocity_size)):
            raise ValueError("pose_size and velocity_size must be positive integers")

    def __call__(self, t: float, x: Array, u: Array) -> Array:
        if not np.isfinite(t):
            raise ValueError("time must be finite")
        x = array(x, "x", (self.pose_size + self.velocity_size,))
        u = vector(u, "u")
        eta, nu = x[: self.pose_size], x[self.pose_size :]
        terms = self.model(t, eta.copy(), nu.copy(), u.copy())
        d = self.velocity_size
        mass = covariance(terms.inertia, d, "inertia", positive=True)
        c = array(terms.coriolis, "C", (d, d))
        damping = array(terms.damping, "D", (d, d))
        restoring = array(terms.restoring, "g", (d,))
        force = array(terms.force, "tau", (d,))
        pose_dot = array(terms.kinematics, "eta_dot", (self.pose_size,))
        acceleration = np.linalg.solve(mass, force - c @ nu - damping @ nu - restoring)
        return array(np.concatenate((pose_dot, acceleration)), "state derivative", x.shape)
