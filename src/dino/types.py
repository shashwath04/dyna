"""Shared contracts. SI units; vectors are rank-one, matrices rank-two."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class Dynamics(Protocol):
    """Return dx/dt (n,) in state-units/s for t [s], x (n,), u (m,)."""

    def __call__(self, t: float, x: Array, u: Array) -> Array: ...


class StateMap(Protocol):
    """Discrete transition or observation; input (n,), output defined by caller."""

    def __call__(self, x: Array) -> Array: ...


class Controller(Protocol):
    """Sampled controller: t [s], state (n,), reference (n,) -> command (m,)."""

    def __call__(self, t: float, state: Array, reference: Array) -> Array: ...


@dataclass(frozen=True)
class Estimate:
    """Mean (n,) and covariance (n,n), with covariance in pairwise state units."""

    mean: Array
    covariance: Array


@dataclass(frozen=True)
class PIDState:
    """Integral contribution [command units] and previous error [error units]."""

    integral: float = 0.0
    previous_error: float = 0.0


@dataclass(frozen=True)
class AttitudeEstimate:
    """Hamilton scalar-first body-to-inertial q (4,), bias (3,) [rad/s], P (6,6).

    Error coordinates are right-multiplicative body-frame angle then gyro bias.
    """

    quaternion: Array
    bias: Array
    covariance: Array
