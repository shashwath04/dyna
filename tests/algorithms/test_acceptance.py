"""Independent analytic acceptance cases; all algorithms must pass."""

import numpy as np
import pytest

from dino import (
    AttitudeEstimate,
    Estimate,
    FixedPIDConfig,
    FPGAState,
    PIDState,
    attitude_control,
    ekf_predict,
    ekf_update,
    fixed_point_pid_step,
    fpga_controller_step,
    integrate_step,
    kalman_predict,
    kalman_update,
    linearize,
    lqr,
    mekf_step,
    pid_step,
)

pytestmark = pytest.mark.student


def test_integrator_constant_derivative():
    actual = integrate_step(
        lambda t, x, u: np.array([2.0]), 0.0, np.array([1.0]), np.array([0.0]), 0.5
    )
    np.testing.assert_allclose(actual, [2.0], atol=1e-12)


def test_pid_saturation_and_backcalculation():
    output, state = pid_step(2.0, PIDState(), 0.1, 2.0, 1.0, 0.5, -1.0, 1.0, 3.0)
    assert output == 1.0
    assert state.integral == pytest.approx(-3.7)
    assert state.previous_error == 2.0


def test_linearization_independent_polynomial():
    a, b = linearize(
        lambda t, x, u: np.array([x[0] ** 2 + 3 * u[0]]), 0.0, np.array([2.0]), np.array([1.0])
    )
    np.testing.assert_allclose(a, [[4.0]], atol=1e-8)
    np.testing.assert_allclose(b, [[3.0]], atol=1e-8)


def test_scalar_lqr():
    # CARE: 2-P^2=0, so positive stabilizing root is sqrt(2).
    k = lqr(np.array([[0.0]]), np.array([[1.0]]), np.array([[2.0]]), np.array([[1.0]]))
    np.testing.assert_allclose(k, [[np.sqrt(2.0)]])


def test_attitude_sign_equivalence():
    q = np.array([np.cos(0.1), np.sin(0.1), 0.0, 0.0])
    desired = np.array([1.0, 0.0, 0.0, 0.0])
    expected = [-4 * np.sin(0.1), 0.0, 0.0]
    for sign in [1, -1]:
        np.testing.assert_allclose(
            attitude_control(sign * q, desired, np.zeros(3), 2.0, 1.0), expected, atol=1e-12
        )


def test_kalman_prediction(scalar_estimate):
    result = kalman_predict(
        scalar_estimate, np.array([[2.0]]), np.array([[1.0]]), np.array([3.0]), np.array([[0.5]])
    )
    np.testing.assert_allclose(result.mean, [3.0])
    np.testing.assert_allclose(result.covariance, [[4.5]])


def test_kalman_update(scalar_estimate):
    result = kalman_update(scalar_estimate, np.array([2.0]), np.array([[1.0]]), np.array([[1.0]]))
    np.testing.assert_allclose(result.mean, [1.0])
    np.testing.assert_allclose(result.covariance, [[0.5]])


def test_ekf_nonlinear_prediction():
    result = ekf_predict(
        Estimate(np.array([2.0]), np.array([[0.25]])),
        lambda x: x * x,
        np.array([[4.0]]),
        np.array([[0.1]]),
    )
    np.testing.assert_allclose(result.mean, [4.0])
    np.testing.assert_allclose(result.covariance, [[4.1]])


def test_ekf_nonlinear_update():
    result = ekf_update(
        Estimate(np.array([2.0]), np.array([[1.0]])),
        np.array([5.0]),
        lambda x: x * x,
        np.array([[4.0]]),
        np.array([[1.0]]),
    )
    np.testing.assert_allclose(result.mean, [2.0 + 4.0 / 17.0])
    np.testing.assert_allclose(result.covariance, [[1.0 / 17.0]])


def test_mekf_stationary_zero_innovation():
    prior = AttitudeEstimate(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.eye(6) * 0.01)
    result = mekf_step(
        prior,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        0.1,
        np.zeros((6, 6)),
        np.eye(3) * 0.01,
    )
    np.testing.assert_allclose(result.quaternion, prior.quaternion, atol=1e-12)
    np.testing.assert_allclose(result.bias, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(result.covariance).min() >= -1e-12


def test_fixed_pid_exact_binary():
    assert fixed_point_pid_step(8, 0, 0, 16, 8, 0, 4, 16, -16, 16) == (8, 4, 8)


def test_fpga_valid_reset_and_hold():
    config = FixedPIDConfig(16, 8, 0, 4, 16, -16, 16)
    initial = FPGAState(2, 3, 4, True)
    assert fpga_controller_step(8, initial, config, reset=True) == FPGAState()
    assert fpga_controller_step(8, initial, config, valid=False) == FPGAState(2, 3, 4, False)
    assert fpga_controller_step(8, FPGAState(), config) == FPGAState(8, 4, 8, True)


@pytest.mark.parametrize("dt", [0.0, -1.0])
def test_integrator_rejects_nonpositive_step(dt):
    with pytest.raises(ValueError):
        integrate_step(lambda t, x, u: x, 0.0, np.ones(1), np.ones(1), dt)
