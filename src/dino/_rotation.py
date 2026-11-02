"""Hamilton scalar-first quaternion and SO(3) primitives; body-to-inertial."""

import numpy as np

from .types import Array


def skew(v: Array) -> Array:
    """Matrix satisfying skew(v) @ w = v cross w."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def multiply(a: Array, b: Array) -> Array:
    """Hamilton product, preserving operand order."""
    return np.concatenate(
        ([a[0] * b[0] - a[1:] @ b[1:]], a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]))
    )


def quaternion_exp(angle: Array) -> Array:
    """Rotation vector [rad] to unit quaternion, with a regular zero-angle limit."""
    theta = float(np.linalg.norm(angle))
    return np.concatenate(([np.cos(theta / 2)], 0.5 * np.sinc(theta / (2 * np.pi)) * angle))


def rotation(q: Array) -> Array:
    """Active body-to-inertial rotation matrix for unit q."""
    v = q[1:]
    return np.asarray(
        (q[0] ** 2 - v @ v) * np.eye(3) + 2 * np.outer(v, v) + 2 * q[0] * skew(v), dtype=float
    )


def right_jacobian(angle: Array) -> Array:
    """SO(3) right Jacobian for covariance reset after exponential injection."""
    theta = float(np.linalg.norm(angle))
    cross = skew(angle)
    if theta < 1e-4:
        a = 0.5 - theta**2 / 24 + theta**4 / 720
        b = 1 / 6 - theta**2 / 120 + theta**4 / 5040
    else:
        a = (1 - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) - a * cross + b * (cross @ cross)
