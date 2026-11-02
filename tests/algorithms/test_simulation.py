"""Ordinary passing tests for the explicitly requested scheduler implementation."""

import numpy as np
import pytest

from dino import simulate

pytestmark = pytest.mark.student


def test_timeline_and_held_input():
    calls = []

    def controller(t, x, reference):
        calls.append(t)
        return np.array([t])

    result = simulate(lambda t, x, u: u, controller, np.zeros(1), np.zeros(1), 0.25, 0.07, 0.1)
    np.testing.assert_allclose(result.time, [0, 0.07, 0.1, 0.17, 0.2, 0.25])
    np.testing.assert_allclose(calls, [0, 0.1, 0.2])
    np.testing.assert_allclose(result.signals["state"][:, 0], [0, 0, 0, 0.007, 0.01, 0.02])
    np.testing.assert_allclose(result.signals["command"][:, 0], [0, 0, 0.1, 0.1, 0.2, 0.2])


@pytest.mark.parametrize("duration, expected", [(0.0, [0.0]), (0.3, [0.0, 0.1, 0.2, 0.3])])
def test_initial_and_final_events(duration, expected):
    calls = []

    def controller(t, x, reference):
        calls.append(t)
        return np.ones(1)

    result = simulate(lambda t, x, u: u, controller, np.zeros(1), np.zeros(1), duration, 0.07, 0.1)
    np.testing.assert_allclose(calls, expected)
    assert result.time[-1] == duration
    assert np.all(np.diff(result.time) > 0)
    np.testing.assert_allclose(result.signals["state"][-1], [duration])


@pytest.mark.parametrize(
    "duration, integration_dt, control_dt",
    [
        (-1.0, 0.1, 0.1),
        (float("nan"), 0.1, 0.1),
        (1.0, 0.0, 0.1),
        (1.0, 0.1, -1.0),
        (1.0, float("inf"), 0.1),
        (1.0, 0.1, float("nan")),
    ],
)
def test_invalid_timing(duration, integration_dt, control_dt):
    with pytest.raises(ValueError):
        simulate(
            lambda t, x, u: x,
            lambda t, x, r: np.ones(1),
            np.zeros(1),
            np.zeros(1),
            duration,
            integration_dt,
            control_dt,
        )


def test_small_period_and_input_preservation():
    x0 = np.array([2.0])
    ref = np.array([3.0])

    def controller(t, x, r):
        x[:] = -99
        r[:] = -99
        return np.ones(1)

    result = simulate(lambda t, x, u: u, controller, x0, ref, 0.05, 0.07, 0.01)
    np.testing.assert_allclose(result.time, np.arange(6) * 0.01)
    np.testing.assert_allclose(result.signals["state"][-1], [2.05])
    np.testing.assert_array_equal(x0, [2.0])
    np.testing.assert_array_equal(ref, [3.0])
