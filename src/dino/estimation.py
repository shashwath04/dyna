"""Linear, extended, and right-multiplicative attitude Kalman filters."""

import math

import numpy as np
from scipy.linalg import expm

from ._rotation import multiply, quaternion_exp, right_jacobian, rotation, skew
from ._validation import array, covariance, unit, vector
from .types import Array, AttitudeEstimate, Estimate, StateMap


def kalman_predict(estimate: Estimate, f: Array, b: Array, u: Array, q: Array) -> Estimate:
    """Discrete prediction x-=F x+B u; P-=F P F.T+Q.
    x (n,), P/F/Q (n,n), B (n,m), u (m,); Q is per-step covariance.
    Preserve input arrays; reject incompatible dimensions, nonfinite data or non-PSD covariance.
    """
    x, p = _prior(estimate)
    f = array(f, "F", (x.size, x.size))
    u = vector(u, "u")
    b = array(b, "B", (x.size, u.size))
    q = covariance(q, x.size, "Q")
    return Estimate(
        array(f @ x + b @ u, "predicted mean"),
        covariance(f @ p @ f.T + q, x.size, "predicted covariance"),
    )


def kalman_update(estimate: Estimate, measurement: Array, h: Array, r: Array) -> Estimate:
    """Measurement z (p,), H (p,n), R (p,p); innovation z-Hx.
    S=H P H.T+R; K=P H.T S^-1; x+=K innovation.
    Joseph covariance: (I-KH)P(I-KH).T+K R K.T.
    Require PSD prior and positive definite R; reject invalid shapes/nonfinite data.
    Covariance units follow state and measurement coordinates; no mutation.
    """
    x, p = _prior(estimate)
    z = vector(measurement, "measurement")
    h = array(h, "H", (z.size, x.size))
    r = covariance(r, z.size, "R", positive=True)
    return _correct(x, p, z - h @ x, h, r)


def ekf_predict(estimate: Estimate, transition: StateMap, jacobian: Array, q: Array) -> Estimate:
    """Discrete EKF prediction x-=f(x), P-=F P F.T+Q.
    F (n,n) is caller-supplied Jacobian at prior mean (n,); Q per-step (n,n).
    Nonlinear transition owns dt and input capture. Reject malformed/nonfinite or non-PSD data.
    """
    x, p = _prior(estimate)
    f = array(jacobian, "Jacobian", (x.size, x.size))
    q = covariance(q, x.size, "Q")
    mean = array(transition(x.copy()), "transition output", x.shape)
    return Estimate(mean, covariance(f @ p @ f.T + q, x.size, "predicted covariance"))


def ekf_update(
    estimate: Estimate, measurement: Array, observation: StateMap, jacobian: Array, r: Array
) -> Estimate:
    """EKF update using innovation z-h(x), H (p,n) at prior mean (n,).
    Gain and Joseph covariance follow kalman_update; z (p,), R (p,p) PD.
    Euclidean measurements only; reject invalid dimensions/nonfinite data; no mutation.
    """
    x, p = _prior(estimate)
    z = vector(measurement, "measurement")
    h = array(jacobian, "Jacobian", (z.size, x.size))
    r = covariance(r, z.size, "R", positive=True)
    predicted = array(observation(x.copy()), "observation output", z.shape)
    return _correct(x, p, z - predicted, h, r)


def mekf_step(
    estimate: AttitudeEstimate,
    gyro: Array,
    measured_body: Array,
    reference_inertial: Array,
    dt: float,
    process_covariance: Array,
    measurement_covariance: Array,
) -> AttitudeEstimate:
    """Propagate body-to-inertial Hamilton q with q_dot=0.5*q tensor [0,gyro-bias].
    Gyro (3,) [rad/s]; one unit direction body/inertial (3,); dt>0 [s].
    Right error q_true=q_nom tensor delta_q, delta_q approximately [1,delta_theta/2].
    Error (delta_theta,delta_bias) covariance (6,6); process covariance (6,6)
    is already discretized; direction measurement covariance (3,3) is PD.
    Predicted direction=R(q).T reference; correct, inject and reset covariance.
    Preserve unit norm and PSD; reject zero/nonunit directions, malformed or nonfinite data.
    One vector leaves rotation about that vector unobservable; do not claim full observability.
    Assumes constant measured rate and nominal bias over dt. Uses exp(F*dt) for
    error propagation and the SO(3) right Jacobian for reset after exponential
    injection. This is a local small-error filter, not a large-error initializer.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    quaternion = unit(estimate.quaternion, 4, "quaternion")
    bias = array(estimate.bias, "bias", (3,))
    p = covariance(estimate.covariance, 6, "P")
    gyro = array(gyro, "gyro", (3,))
    measured = unit(measured_body, 3, "measured_body")
    reference = unit(reference_inertial, 3, "reference_inertial")
    q = covariance(process_covariance, 6, "Q")
    r = covariance(measurement_covariance, 3, "R", positive=True)

    # Constant bias-corrected body rate: exact nominal rotation over this sample.
    omega = gyro - bias
    predicted_q = multiply(quaternion, quaternion_exp(omega * dt))
    predicted_q /= np.linalg.norm(predicted_q)
    # Right error dynamics: delta_theta_dot = -[omega]x delta_theta-delta_bias.
    generator = np.zeros((6, 6))
    generator[:3, :3] = -skew(omega)
    generator[:3, 3:] = -np.eye(3)
    transition = np.asarray(expm(generator * dt), dtype=float)
    predicted_p = covariance(transition @ p @ transition.T + q, 6, "predicted covariance")
    direction = rotation(predicted_q).T @ reference
    h = np.zeros((3, 6))
    h[:, :3] = skew(direction)
    correction = _correct(np.zeros(6), predicted_p, measured - direction, h, r)

    # Inject the estimated right error, then express covariance in the new chart.
    angle = correction.mean[:3]
    corrected_q = multiply(predicted_q, quaternion_exp(angle))
    corrected_q /= np.linalg.norm(corrected_q)
    reset = np.eye(6)
    reset[:3, :3] = right_jacobian(angle)
    corrected_p = covariance(reset @ correction.covariance @ reset.T, 6, "reset covariance")
    return AttitudeEstimate(
        corrected_q, array(bias + correction.mean[3:], "corrected bias"), corrected_p
    )


def _prior(estimate: Estimate) -> tuple[Array, Array]:
    """Copy and validate a Euclidean Gaussian state."""
    mean = vector(estimate.mean, "mean")
    return mean, covariance(estimate.covariance, mean.size, "P")


def _correct(mean: Array, p: Array, innovation: Array, h: Array, r: Array) -> Estimate:
    """Shared innovation solve and Joseph covariance update; no explicit inverse."""
    cross = p @ h.T
    innovation_cov = h @ cross + r
    try:
        gain = np.linalg.solve(innovation_cov, cross.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular innovation covariance") from exc
    complement = np.eye(mean.size) - gain @ h
    updated_p = complement @ p @ complement.T + gain @ r @ gain.T
    return Estimate(
        array(mean + gain @ innovation, "updated mean"),
        covariance(updated_p, mean.size, "updated covariance"),
    )
