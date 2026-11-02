"""Gaussian conditioning and independent attitude/covariance reference cases."""

import numpy as np
import pytest

from dino import (
    AttitudeEstimate,
    Estimate,
    ekf_predict,
    ekf_update,
    kalman_predict,
    kalman_update,
    mekf_step,
)

pytestmark = pytest.mark.student


def test_correlated_gaussian_conditioning():
    prior = Estimate(np.zeros(2), np.array([[2.0, 1.0], [1.0, 2.0]]))
    result = kalman_update(prior, np.array([3.0]), np.array([[1.0, 0.0]]), np.eye(1))
    np.testing.assert_allclose(result.mean, [2, 1])
    np.testing.assert_allclose(result.covariance, [[2 / 3, 1 / 3], [1 / 3, 5 / 3]])
    np.testing.assert_array_equal(prior.mean, np.zeros(2))
    np.testing.assert_array_equal(prior.covariance, [[2, 1], [1, 2]])


def test_ekf_linear_equivalence():
    prior = Estimate(np.array([2.0, 3.0]), np.array([[2.0, 0.5], [0.5, 1.0]]))
    f, b, u, q = np.array([[1.0, 0.1], [0.0, 1.0]]), np.ones((2, 1)), np.ones(1), np.eye(2) * 0.01
    kf = kalman_predict(prior, f, b, u, q)
    ekf = ekf_predict(prior, lambda x: f @ x + b @ u, f, q)
    np.testing.assert_allclose(ekf.mean, kf.mean)
    np.testing.assert_allclose(ekf.covariance, kf.covariance)
    z, h, r = np.array([2.0]), np.array([[1.0, 0.0]]), np.eye(1) * 0.2
    kf, ekf = kalman_update(kf, z, h, r), ekf_update(ekf, z, lambda x: h @ x, h, r)
    np.testing.assert_allclose(ekf.mean, kf.mean)
    np.testing.assert_allclose(ekf.covariance, kf.covariance)


@pytest.mark.parametrize(
    "p",
    [np.diag([1.0, -1.0]), np.array([[1.0, 1.0], [0.0, 1.0]]), np.full((2, 2), np.nan), np.eye(3)],
)
def test_invalid_prior_covariance(p):
    with pytest.raises(ValueError):
        kalman_update(Estimate(np.ones(2), p), np.ones(1), np.ones((1, 2)), np.eye(1))


def test_invalid_measurement_noise_and_callback_shape():
    prior = Estimate(np.ones(1), np.eye(1))
    with pytest.raises(ValueError):
        kalman_update(prior, np.ones(1), np.eye(1), np.zeros((1, 1)))
    with pytest.raises(ValueError):
        ekf_predict(prior, lambda x: np.ones(2), np.eye(1), np.eye(1))
    with pytest.raises(ValueError):
        ekf_update(prior, np.ones(1), lambda x: np.ones((1, 1)), np.eye(1), np.eye(1))


def test_mekf_analytic_stationary_covariance():
    # With zero nominal rate: theta-=theta-dt*bias. Measurement of z observes
    # the two transverse angles with scalar noise r, leaving yaw unobserved.
    a, b, dt, r = 0.02, 0.03, 0.2, 0.04
    prior = AttitudeEstimate(
        np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.diag([a] * 3 + [b] * 3)
    )
    out = mekf_step(
        prior,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        dt,
        np.zeros((6, 6)),
        np.eye(3) * r,
    )
    angle_var, cross = a + dt**2 * b, -dt * b
    expected = np.zeros((6, 6))
    for axis in range(3):
        if axis < 2:
            expected[axis, axis] = angle_var * r / (angle_var + r)
            expected[axis + 3, axis + 3] = b - cross**2 / (angle_var + r)
            expected[axis, axis + 3] = expected[axis + 3, axis] = cross * r / (angle_var + r)
        else:
            expected[axis, axis] = angle_var
            expected[axis + 3, axis + 3] = b
            expected[axis, axis + 3] = expected[axis + 3, axis] = cross
    np.testing.assert_allclose(out.covariance, expected, atol=1e-14)


def test_mekf_single_axis_rotation_and_sign():
    dt, rate = 0.2, 0.7
    expected = np.array([np.cos(rate * dt / 2), 0.0, 0.0, np.sin(rate * dt / 2)])
    for sign in [-1, 1]:
        prior = AttitudeEstimate(
            sign * np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.eye(6) * 0.01
        )
        out = mekf_step(
            prior,
            np.array([0.0, 0.0, rate]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            dt,
            np.zeros((6, 6)),
            np.eye(3) * 0.01,
        )
        np.testing.assert_allclose(out.quaternion, sign * expected, atol=1e-13)


def test_mekf_measurement_corrects_roll_and_estimates_bias():
    theta = 0.1
    prior = AttitudeEstimate(
        np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0]), np.zeros(3), np.eye(6) * 0.1
    )
    for _ in range(200):
        prior = mekf_step(
            prior,
            np.array([0.02, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
            0.05,
            np.eye(6) * 1e-8,
            np.eye(3) * 0.001,
        )
    assert abs(prior.quaternion[1]) < 0.001
    assert prior.bias[0] == pytest.approx(0.02, abs=0.001)
    assert np.linalg.norm(prior.quaternion) == pytest.approx(1.0)
    assert np.linalg.eigvalsh(prior.covariance).min() >= -1e-12


def test_mekf_rejects_nonunit_direction():
    with pytest.raises(ValueError):
        mekf_step(
            AttitudeEstimate(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.eye(6)),
            np.zeros(3),
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            0.1,
            np.eye(6),
            np.eye(3),
        )


def test_mekf_nonidentity_frame_and_input_preservation():
    # Known 90-degree yaw: inertial +x appears as body -y.
    q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    prior = AttitudeEstimate(q.copy(), np.zeros(3), np.eye(6) * 0.01)
    out = mekf_step(
        prior,
        np.zeros(3),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        0.1,
        np.zeros((6, 6)),
        np.eye(3) * 0.01,
    )
    np.testing.assert_allclose(out.quaternion, q, atol=1e-12)
    np.testing.assert_array_equal(prior.quaternion, q)
    np.testing.assert_array_equal(prior.covariance, np.eye(6) * 0.01)


def test_rotation_reset_jacobian_against_independent_rotation_library():
    # Differentiate Log(Exp(-a) Exp(a+epsilon)) at epsilon=0. This checks
    # the reset's sign and chart convention independently of quaternion code.
    from scipy.spatial.transform import Rotation

    from dino._rotation import right_jacobian

    a = np.array([0.2, -0.1, 0.3])
    numerical = np.empty((3, 3))
    base_inverse = Rotation.from_rotvec(-a)
    for j in range(3):
        perturb = np.zeros(3)
        perturb[j] = 1e-6
        plus = (base_inverse * Rotation.from_rotvec(a + perturb)).as_rotvec()
        minus = (base_inverse * Rotation.from_rotvec(a - perturb)).as_rotvec()
        numerical[:, j] = (plus - minus) / (2e-6)
    np.testing.assert_allclose(right_jacobian(a), numerical, atol=1e-9)
    np.testing.assert_allclose(right_jacobian(np.zeros(3)), np.eye(3), atol=1e-15)
