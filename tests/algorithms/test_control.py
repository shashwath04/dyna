"""Analytic control references and numerical contract checks."""

import numpy as np
import pytest

from dino import PIDState, attitude_control, integrate_step, linearize, lqr, pid_step

pytestmark = pytest.mark.student


def test_euler_convergence_and_input_isolation():
    errors = []
    for count in (10, 20, 40):
        x = np.ones(1)
        for i in range(count):
            x = integrate_step(lambda t, x, u: -x, i / count, x, np.zeros(1), 1 / count)
        errors.append(abs(x[0] - np.exp(-1)))
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 2.0, rtol=0.06)
    x, u = np.ones(1), np.ones(1)

    def mutating(t, state, command):
        state[:] = 7
        command[:] = 8
        return np.ones(1)

    np.testing.assert_allclose(integrate_step(mutating, 0, x, u, 0.1), [1.1])
    np.testing.assert_array_equal(x, [1])
    np.testing.assert_array_equal(u, [1])


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf])
def test_integrator_bad_dt(dt):
    with pytest.raises(ValueError):
        integrate_step(lambda t, x, u: x, 0, np.ones(1), np.ones(1), dt)


@pytest.mark.parametrize("derivative", [np.ones((1, 1)), np.ones(2), np.array([np.nan])])
def test_integrator_bad_derivative(derivative):
    with pytest.raises(ValueError):
        integrate_step(lambda t, x, u: derivative, 0, np.ones(1), np.ones(1), 0.1)


def test_pid_sequence_and_tracking_recovery():
    command, state = pid_step(1, PIDState(), 0.1, 2, 3, 0, -10, 10)
    assert command == 2
    assert state.integral == pytest.approx(0.3)
    command, state = pid_step(1, state, 0.1, 2, 3, 0, -10, 10)
    assert command == pytest.approx(2.3)
    assert state.integral == pytest.approx(0.6)
    # Back-calculation drives a wound-up integral toward the output limit.
    _, corrected = pid_step(0, PIDState(10, 0), 0.1, 0, 0, 0, -1, 1, 2)
    assert corrected.integral == pytest.approx(8.2)


@pytest.mark.parametrize(
    "kwargs", [{"dt": 0}, {"lower": 2}, {"tracking_gain": -1}, {"error": np.nan}, {"upper": np.inf}]
)
def test_pid_invalid(kwargs):
    values = {
        "error": 0.0,
        "state": PIDState(),
        "dt": 0.1,
        "kp": 1.0,
        "ki": 1.0,
        "kd": 0.0,
        "lower": -1.0,
        "upper": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        pid_step(**values)


def test_multivariable_linearization():
    x, u = np.array([2.0, 3.0]), np.array([4.0, 5.0])
    a, b = linearize(
        lambda t, x, u: np.array([x[0] * x[1] + u[0], x[0] + u[1] ** 2 + t]), 0.5, x, u
    )
    np.testing.assert_allclose(a, [[3, 2], [1, 0]], atol=1e-8)
    np.testing.assert_allclose(b, [[1, 0], [0, 10]], atol=1e-8)
    np.testing.assert_array_equal(x, [2, 3])
    np.testing.assert_array_equal(u, [4, 5])


def test_double_integrator_lqr_and_care():
    a, b = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])
    q, r = np.eye(2), np.eye(1)
    k = lqr(a, b, q, r)
    np.testing.assert_allclose(k, [[1.0, np.sqrt(3)]], atol=1e-12)
    # Independent closed-form CARE solution for the unit-weight double integrator.
    p = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    np.testing.assert_allclose(a.T @ p + p @ a - p @ b @ b.T @ p + q, np.zeros((2, 2)), atol=1e-12)
    assert np.all(np.linalg.eigvals(a - b @ k).real < 0)


def test_lqr_stable_uncontrolled_mode():
    np.testing.assert_allclose(
        lqr(-np.eye(2), np.array([[1.0], [0.0]]), np.diag([1.0, 0.0]), np.eye(1)),
        [[np.sqrt(2) - 1, 0]],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "a,b,q,r",
    [
        (np.eye(1), np.zeros((1, 1)), np.eye(1), np.eye(1)),
        (np.zeros((1, 1)), np.eye(1), np.eye(1), np.zeros((1, 1))),
        (np.zeros((1, 1)), np.eye(1), -np.eye(1), np.eye(1)),
        (np.zeros((2, 2)), np.ones((2, 1)), np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(1)),
    ],
)
def test_invalid_lqr(a, b, q, r):
    with pytest.raises(ValueError):
        lqr(a, b, q, r)


def test_attitude_damping_and_half_turn_tie():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(attitude_control(identity, identity, np.ones(3), 2, 3), [-3] * 3)
    for sign in (-1, 1):
        np.testing.assert_allclose(
            attitude_control(sign * np.array([0.0, -1.0, 0.0, 0.0]), identity, np.zeros(3), 2, 0),
            [-4, 0, 0],
        )
    with pytest.raises(ValueError):
        attitude_control(identity * 2, identity, np.zeros(3), 1, 1)


def test_lqr_multiaxis_against_independent_riccati_solver():
    from scipy.linalg import solve_continuous_are

    a = np.array([[0.0, 1.0, 0.0], [-2.0, -0.3, 0.5], [0.0, 0.0, -0.4]])
    b = np.array([[0.0, 0.0], [1.0, 0.2], [0.0, 1.0]])
    q, r = np.diag([4.0, 2.0, 1.0]), np.diag([0.5, 2.0])
    p = solve_continuous_are(a, b, q, r)
    expected = np.linalg.solve(r, b.T @ p)
    np.testing.assert_allclose(lqr(a, b, q, r), expected, atol=1e-11)
