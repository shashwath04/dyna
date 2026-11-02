"""Forward Euler integration and event-aligned sampled-control scheduling."""

import math

import numpy as np

from ._validation import array, vector
from .logging import SignalLogger, SignalSpec, SimulationResult
from .types import Array, Controller, Dynamics


def integrate_step(dynamics: Dynamics, t: float, x: Array, u: Array, dt: float) -> Array:
    """Advance dx/dt=f(t,x,u) with one forward Euler step, holding u constant.

    Time and dt are seconds; x (n,) and u (m,) are finite real vectors.
    Returns x + dt*f(t,x,u) (n,) in state units, without mutating inputs.
    One-step error O(dt**2), global error O(dt) for smooth, stable trajectories.
    Reject nonfinite data, invalid derivative shapes, dt<=0 and overflow.
    """
    if not math.isfinite(t) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("t must be finite and dt finite and positive")
    x = vector(x, "x")
    u = vector(u, "u")
    derivative = array(dynamics(t, x.copy(), u.copy()), "derivative", x.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        return array(x + dt * derivative, "next state", x.shape)


def simulate(
    dynamics: Dynamics,
    controller: Controller,
    x0: Array,
    reference: Array,
    duration: float,
    integration_dt: float,
    control_dt: float,
) -> SimulationResult:
    """Schedule continuous plant and discrete control from zero through duration.

    All times are seconds. x0/reference (n,), command (m,). Controller runs at
    zero and multiples of control_dt, including the final time if coincident.
    Hold the old command through each integration interval; update after reaching
    a control event. Log state (N,n) and held command (N,m) after each endpoint.
    Restart the maximum integration step after every endpoint. Never overshoot.
    Final/control times within four floating-point spacings count as coincident.
    Duration=0 records one initial sample. Reject nonfinite/negative duration,
    nonfinite/nonpositive periods, inconsistent shapes or inability to advance time.
    Physical component units belong to the model; log labels indicate mixed SI.
    """
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    if not math.isfinite(integration_dt) or integration_dt <= 0:
        raise ValueError("integration_dt must be finite and positive")
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    x = vector(x0, "x0")
    reference = array(reference, "reference", x.shape)

    def command_at(time: float) -> Array:
        return vector(controller(time, x.copy(), reference.copy()), "command")

    t = 0.0
    held_command = command_at(t)
    logger = SignalLogger(
        {
            "state": SignalSpec(x.size, "component-dependent SI"),
            "command": SignalSpec(held_command.size, "component-dependent SI"),
        }
    )
    logger.append(t, {"state": x, "command": held_command})
    control_index = 1
    next_control_time = control_dt

    while t < duration:
        if math.isfinite(next_control_time) and math.isclose(
            next_control_time,
            duration,
            rel_tol=0.0,
            abs_tol=4 * max(math.ulp(next_control_time), math.ulp(duration)),
        ):
            next_control_time = duration
        # Stop at whichever boundary comes first: integration, control, or finish.
        stop_time = min(t + integration_dt, next_control_time, duration)
        step_dt = stop_time - t
        if step_dt <= 0:
            raise ValueError("time cannot advance at this floating-point resolution")
        x = integrate_step(dynamics, t, x, held_command, step_dt)
        t = stop_time
        # The old command drove the interval. The new command starts here.
        if t == next_control_time:
            command = command_at(t)
            if command.shape != held_command.shape:
                raise ValueError("command shape must remain constant")
            held_command = command
            control_index += 1
            next_control_time = control_index * control_dt
        logger.append(t, {"state": x, "command": held_command})
    return logger.result()


def rk4_step(dynamics: Dynamics, t: float, x: Array, u: Array, dt: float) -> Array:
    """Classical RK4 with held u, local error O(dt**5), global O(dt**4).

    Same shapes/validation as integrate_step. Dynamics are evaluated at four
    intermediate states/times; discontinuities must be split by the caller.
    Does not automatically normalize quaternion or constrained states.
    """
    if not math.isfinite(t) or not math.isfinite(dt) or dt <= 0 or not math.isfinite(t + dt):
        raise ValueError("time must be finite and dt finite and positive")
    x, u = vector(x, "x"), vector(u, "u")

    def evaluate(time: float, state: Array) -> Array:
        return array(dynamics(time, state.copy(), u.copy()), "derivative", x.shape)

    k1 = evaluate(t, x)
    k2 = evaluate(t + dt / 2, x + dt * k1 / 2)
    k3 = evaluate(t + dt / 2, x + dt * k2 / 2)
    k4 = evaluate(t + dt, x + dt * k3)
    return array(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, "next state", x.shape)
