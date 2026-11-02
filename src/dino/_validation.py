"""Internal numerical contracts shared by control and estimation."""

import numpy as np

from .types import Array


def array(value: Array, name: str, shape: tuple[int, ...] | None = None) -> Array:
    """Copy real finite data, optionally enforcing an exact shape."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def vector(value: Array, name: str) -> Array:
    """Require a nonempty rank-one real vector."""
    result = array(value, name)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    return result


def covariance(value: Array, size: int, name: str, *, positive: bool = False) -> Array:
    """Validate symmetry and PSD (relative spectral tolerance 1e-10), or strict PD.

    Only roundoff-sized negative PSD eigenvalues are projected to zero.
    """
    result = array(value, name, (size, size))
    scale = max(float(np.linalg.norm(result, ord=2)), np.finfo(float).tiny)
    if np.max(np.abs(result - result.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    result = (result + result.T) / 2
    eigenvalues, basis = np.linalg.eigh(result)
    if positive:
        if eigenvalues[0] <= 0:
            raise ValueError(f"{name} must be positive definite")
    elif eigenvalues[0] < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    elif eigenvalues[0] < 0:
        result = (basis * np.maximum(eigenvalues, 0)) @ basis.T
    return result


def unit(value: Array, size: int, name: str) -> Array:
    """Require unit length within absolute tolerance 1e-8; normalize roundoff."""
    result = array(value, name, (size,))
    norm = float(np.linalg.norm(result))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must have unit norm")
    return result / norm
