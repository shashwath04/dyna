"""Independent manufactured marine/mass-matrix equations and solver checks."""

import numpy as np
import pytest

from dino import FossenDynamics, FossenTerms, MassMatrixDynamics, rk4_step

pytestmark = pytest.mark.student


def test_fossen_terms_with_coupled_inertia():
    mass = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    c = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    d, g = np.diag([1.0, 2.0, 3.0]), np.ones(3)
    nu, expected = np.array([2.0, 3.0, 4.0]), np.array([1.0, -2.0, 0.5])
    force = mass @ expected + c @ nu + d @ nu + g

    def terms(t, eta, velocity, command):
        return FossenTerms(mass, c, d, g, command, np.array([4.0, 5.0, 6.0]))

    dynamics = FossenDynamics(terms)
    state = np.concatenate((np.zeros(3), nu))
    np.testing.assert_allclose(dynamics(0.0, state, force), [4.0, 5.0, 6.0, 1.0, -2.0, 0.5])
    np.testing.assert_array_equal(state[3:], nu)


@pytest.mark.parametrize("pose,velocity", [(3, 3), (6, 6), (7, 6)])
def test_custom_layouts(pose, velocity):
    def terms(t, eta, nu, command):
        return FossenTerms(
            np.eye(velocity),
            np.zeros((velocity, velocity)),
            np.zeros((velocity, velocity)),
            np.zeros(velocity),
            command,
            np.ones(pose),
        )

    dynamics = FossenDynamics(terms, pose, velocity)
    out = dynamics(0.0, np.zeros(pose + velocity), np.ones(velocity) * 2)
    np.testing.assert_allclose(out, np.concatenate((np.ones(pose), np.ones(velocity) * 2)))


def test_state_dependent_mass_matrix_and_singular_rejection():
    dynamics = MassMatrixDynamics(lambda t, x, u: (np.diag(1 + x * x), (1 + x * x) * (-x + u)))
    np.testing.assert_allclose(dynamics(0.0, np.array([2.0, 3.0]), np.ones(2)), [-1.0, -2.0])
    with pytest.raises(ValueError):
        MassMatrixDynamics(lambda t, x, u: (np.zeros((1, 1)), np.ones(1)))(
            0, np.ones(1), np.ones(1)
        )


def test_rk4_convergence_and_time_dependence():
    errors = []
    for count in [4, 8, 16]:
        x = np.ones(1)
        for i in range(count):
            x = rk4_step(lambda t, x, u: -x, i / count, x, np.ones(1), 1 / count)
        errors.append(abs(x[0] - np.exp(-1)))
    assert 14 < errors[1] / errors[2] < 18
    np.testing.assert_allclose(
        rk4_step(lambda t, x, u: np.array([t**3]), 0.0, np.zeros(1), np.ones(1), 1.0),
        [0.25],
        atol=1e-14,
    )


def test_bad_inertia():
    def terms(t, eta, nu, command):
        return FossenTerms(
            -np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros(3)
        )

    with pytest.raises(ValueError):
        FossenDynamics(terms)(0.0, np.zeros(6), np.ones(1))


def test_boat_example_pure_yaw_matches_closed_form():
    import runpy
    from pathlib import Path

    from dino import ServoPacket

    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "examples/ardurover_boat.py")
    )
    runner = namespace["make_runner"]()
    pwm = [1500] * 16
    pwm[0], pwm[2] = 2000, 1000  # equal and opposite thrust: pure positive yaw
    for count in range(50):
        runner.handle(ServoPacket(50, count, tuple(pwm)))
    # 15*r_dot + 5*r = 24 N m, r(0)=0. This case has no Coriolis coupling.
    time = runner.time
    expected_rate = (24 / 5) * (1 - np.exp(-time / 3))
    expected_yaw = (24 / 5) * (time - 3 * (1 - np.exp(-time / 3)))
    np.testing.assert_allclose(runner.state, [0, 0, expected_yaw, 0, 0, expected_rate], atol=1e-11)
