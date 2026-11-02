"""Sampled PID, central-difference linearization, LQR, and quaternion feedback."""

import math

import numpy as np
from scipy.linalg import schur

from ._rotation import multiply
from ._validation import array, covariance, unit, vector
from .types import Array, Dynamics, PIDState


def pid_step(
    error: float,
    state: PIDState,
    dt: float,
    kp: float,
    ki: float,
    kd: float,
    lower: float,
    upper: float,
    tracking_gain: float = 0.0,
) -> tuple[float, PIDState]:
    """Discrete PID with backward difference and back-calculation anti-windup.
    e in error units; dt [s]; I is command contribution. D=kd*(e-e_prev)/dt.
    v=kp*e+I+D; u=clip(v,lower,upper).
    I_next=I+dt*(ki*e+tracking_gain*(u-v)). Return u and next state.
    Reject dt<=0, reversed bounds, negative tracking_gain and nonfinite inputs.
    Gains: kp [command/error], ki [command/error/s], kd [command*s/error].
    """
    values = (
        error,
        state.integral,
        state.previous_error,
        dt,
        kp,
        ki,
        kd,
        lower,
        upper,
        tracking_gain,
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("PID inputs must be finite")
    if dt <= 0 or lower > upper or tracking_gain < 0:
        raise ValueError("invalid PID period, bounds, or tracking gain")
    unsaturated = kp * error + state.integral + kd * (error - state.previous_error) / dt
    command = float(np.clip(unsaturated, lower, upper))
    integral = state.integral + dt * (ki * error + tracking_gain * (command - unsaturated))
    if not math.isfinite(unsaturated) or not math.isfinite(integral):
        raise ValueError("PID arithmetic overflow")
    return command, PIDState(integral, error)


def linearize(
    dynamics: Dynamics, t: float, x: Array, u: Array, epsilon: float = 1e-6
) -> tuple[Array, Array]:
    """Return A=df/dx (n,n), B=df/du (n,m) at (t [s],x,u).
    Use central differences with positive absolute epsilon in each coordinate's units.
    A units are state_i/state_j/s; B state_i/input_j/s.
    Reject invalid shapes, nonfinite inputs and epsilon<=0; no mutation.
    """
    if not math.isfinite(t) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("time must be finite and epsilon finite and positive")
    x = vector(x, "x")
    u = vector(u, "u")

    def evaluate(state: Array, command: Array) -> Array:
        return array(dynamics(t, state.copy(), command.copy()), "derivative", x.shape)

    a = np.empty((x.size, x.size))
    b = np.empty((x.size, u.size))
    for values, output, is_state in ((x, a, True), (u, b, False)):
        for j in range(values.size):
            plus, minus = values.copy(), values.copy()
            plus[j] += epsilon
            minus[j] -= epsilon
            if plus[j] == values[j] or minus[j] == values[j]:
                raise ValueError("epsilon is below the coordinate's floating-point resolution")
            if is_state:
                output[:, j] = (evaluate(plus, u) - evaluate(minus, u)) / (2 * epsilon)
            else:
                output[:, j] = (evaluate(x, plus) - evaluate(x, minus)) / (2 * epsilon)
    return array(a, "A"), array(b, "B")


def lqr(a: Array, b: Array, q: Array, r: Array) -> Array:
    """Continuous infinite-horizon LQR, u=-Kx; return K (m,n).
    A (n,n), B (n,m), Q (n,n) PSD, R (m,m) positive definite.
    Solve A.T P+P A-P B R^-1 B.T P+Q=0, K=R^-1 B.T P.
    Cost integral x.T Q x+u.T R u dt. Reject incompatible/nonfinite data,
    asymmetric weights, non-stabilizable systems or absence of stabilizing solution.
    Uses the stable Hamiltonian Schur subspace; verifies CARE residual and Hurwitz poles.
    """
    a = array(a, "A")
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be nonempty and square")
    n = a.shape[0]
    b = array(b, "B")
    if b.ndim != 2 or b.shape[0] != n or b.shape[1] == 0:
        raise ValueError("B must have shape (n,m), m>0")
    q = covariance(q, n, "Q")
    r = covariance(r, b.shape[1], "R", positive=True)
    # CARE's stabilizing solution is the graph of the Hamiltonian's stable
    # invariant subspace. Ordered Schur avoids choosing individual eigenvectors.
    g = b @ np.linalg.solve(r, b.T)
    hamiltonian = np.block([[a, -g], [-q, -a.T]])
    try:
        _, basis, count = schur(hamiltonian, output="complex", sort=lambda z: z.real < 0)
        if count != n:
            raise ValueError("Hamiltonian has no n-dimensional stable subspace")
        upper, lower = basis[:n, :n], basis[n:, :n]
        solution = np.linalg.solve(upper.T, lower.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("no numerically resolvable stabilizing CARE solution") from exc
    if np.max(np.abs(solution.imag)) > 1e-8 * max(1.0, float(np.linalg.norm(solution))):
        raise ValueError("CARE solution is not real")
    p = covariance(np.asarray(solution.real, dtype=float), n, "CARE solution")
    residual = a.T @ p + p @ a - p @ g @ p + q
    scale = max(
        1.0,
        float(
            np.linalg.norm(a.T @ p)
            + np.linalg.norm(p @ a)
            + np.linalg.norm(p @ g @ p)
            + np.linalg.norm(q)
        ),
    )
    k = np.linalg.solve(r, b.T @ p)
    if np.linalg.norm(residual) > 1e-8 * scale or np.any(np.linalg.eigvals(a - b @ k).real >= 0):
        raise ValueError("CARE solution does not satisfy residual/stability checks")
    return array(k, "K")


def attitude_control(
    quaternion: Array, desired: Array, omega: Array, kp: float, kd: float
) -> Array:
    """Return body torque (3,) [N m] with scalar-first Hamilton body-to-inertial q.
    q_error=conjugate(desired) tensor quaternion; shortest arc uses nonnegative scalar.
    torque=-kp*2*vector(q_error)-kd*omega; omega (3,) [rad/s].
    Desired rate is zero; gains isotropic. At scalar=0 choose first nonzero vector
    component positive. Require unit quaternions, finite data and nonnegative gains.
    """
    quaternion = unit(quaternion, 4, "quaternion")
    desired = unit(desired, 4, "desired")
    omega = array(omega, "omega", (3,))
    if not math.isfinite(kp) or not math.isfinite(kd) or kp < 0 or kd < 0:
        raise ValueError("attitude gains must be finite and nonnegative")
    conjugate = desired.copy()
    conjugate[1:] *= -1
    error = multiply(conjugate, quaternion)
    if error[0] < 0:
        error = -error
    elif error[0] == 0:
        # Resolve the exact 180-degree ambiguity consistently for q and -q.
        for component in error[1:]:
            if component != 0:
                if component < 0:
                    error = -error
                break
    return array(-2 * kp * error[1:] - kd * omega, "torque")
