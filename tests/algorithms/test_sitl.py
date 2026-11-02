"""Lockstep frame sequencing and real UDP runner integration with custom physics."""

import json
import socket

import numpy as np
import pytest

from dino import ArduPilotUDP, FrameSequenceError, SensorFrame, ServoPacket, SITLRunner, rk4_step

pytestmark = pytest.mark.student


def make_runner(max_step=0.007, on_reset=None):
    return SITLRunner(
        lambda t, x, u: u,
        np.zeros(1),
        lambda packet: np.array([packet.pwm[0] / 1000]),
        lambda t, x, u: SensorFrame(
            t,
            np.array([x[0], 0.0, 0.0]),
            np.array([u[0], 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            np.array([0.0, 0.0, -9.80665]),
        ),
        max_step=max_step,
        stepper=rk4_step,
        on_reset=on_reset,
    )


def packet(count, rate=50, channels=16, pwm=1500):
    return ServoPacket(rate, count, (pwm,) * channels)


def test_response_timing_hold_and_duplicate_cache():
    runner = make_runner()
    first = runner.handle(packet(0))
    assert json.loads(first)["timestamp"] == 0.02
    np.testing.assert_allclose(runner.state, [0.03])
    assert runner.handle(packet(0)) is first
    assert runner.result().time.size == 1
    runner.handle(packet(1, pwm=1000))
    np.testing.assert_allclose(runner.state, [0.05])
    np.testing.assert_allclose(runner.result().time, [0.02, 0.04])
    np.testing.assert_allclose(runner.result().signals["command"][:, 0], [1.5, 1.0])


def test_counter_restart_rollover_and_rate_change():
    resets = []
    runner = make_runner(on_reset=lambda: resets.append(True))
    runner.handle(packet(0xFFFFFFFF))
    runner.handle(packet(0))  # uint32 rollover, not restart
    assert runner.restarts == 0
    runner.handle(packet(1, rate=100, channels=32))
    assert runner.time == pytest.approx(0.05)
    runner.handle(packet(0))
    assert runner.restarts == 1 and len(resets) == 1
    assert runner.time == 0.02
    assert runner.result().time.size == 1


def test_gaps_backwards_and_changed_duplicates_do_not_advance():
    runner = make_runner()
    runner.handle(packet(4))
    for invalid in [packet(6), packet(3), packet(4, rate=100)]:
        with pytest.raises(FrameSequenceError):
            runner.handle(invalid)
        assert runner.time == 0.02
        np.testing.assert_allclose(runner.state, [0.03])


def test_observer_timestamp_validation():
    runner = make_runner()
    runner.observe = lambda t, x, u: SensorFrame(
        0.0, np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)
    )
    with pytest.raises(ValueError):
        runner.handle(packet(0))
    assert runner.time == 0
    np.testing.assert_array_equal(runner.state, [0])


def test_runner_actual_udp_transaction():
    runner = make_runner()
    with (
        ArduPilotUDP(port=0) as endpoint,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
    ):
        sender.settimeout(1.0)
        for count in [0, 1, 1, 2]:
            sender.sendto(packet(count).encode(), endpoint.address)
            runner.serve_once(endpoint)
            response = json.loads(sender.recvfrom(65535)[0])
            assert response["timestamp"] == pytest.approx((count + 1) / 50)
        assert runner.result().time.size == 3
        np.testing.assert_allclose(runner.state, [0.09])
