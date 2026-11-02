"""Wire-format fixtures and real loopback UDP transactions."""

import json
import socket
import struct

import numpy as np
import pytest

from dino import ArduPilotUDP, SensorFrame, ServoPacket, body_to_sensor_frame


def stationary_frame(t=0.0):
    return SensorFrame(
        t,
        np.zeros(3),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, -9.80665]),
    )


@pytest.mark.parametrize("channels,magic", [(16, 18458), (32, 29569)])
def test_protocol_fixture(channels, magic):
    # Independent binary fixture, not only an encode/decode round trip.
    wire = struct.pack("<HHI", magic, 50, 0x12345678) + struct.pack(
        f"<{channels}H", *range(channels)
    )
    packet = ServoPacket.decode(wire)
    assert packet.frame_count == 0x12345678
    assert packet.frame_rate == 50
    assert packet.pwm == tuple(range(channels))
    assert packet.encode() == wire


@pytest.mark.parametrize(
    "wire",
    [
        b"",
        b"x",
        b"\0" * 40,
        struct.pack("<HHI16H", 18458, 0, 1, *([1500] * 16)),
        struct.pack("<HHI16H", 18458, 50, 1, *([1500] * 16)) + b"x",
        struct.pack("<HHI16H", 29569, 50, 1, *([1500] * 16)),
    ],
)
def test_bad_packets(wire):
    with pytest.raises(ValueError):
        ServoPacket.decode(wire)


def test_sensor_wire_and_finite_validation():
    wire = stationary_frame().to_json()
    assert wire.startswith(b"\n") and wire.endswith(b"\n")
    data = json.loads(wire)
    assert data["imu"]["accel_body"] == [0, 0, -9.80665]
    assert data["quaternion"] == [1, 0, 0, 0]
    assert data["no_time_sync"] is False and data["no_lockstep"] is False
    with pytest.raises(ValueError):
        stationary_frame(np.nan).to_json()
    with pytest.raises(ValueError):
        SensorFrame(0, np.zeros(3), np.zeros(3), np.ones(4), np.zeros(3), np.zeros(3)).to_json()


def test_body_frames_gravity_and_transport_acceleration():
    q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])  # body forward = east
    frame = body_to_sensor_frame(
        1.0,
        np.zeros(3),
        q,
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 3.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(frame.velocity_ned, [0.0, 2.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(frame.accel_body, [1.0, 6.0, -9.80665])
    # A stationary tilted craft senses gravity expressed in its rotated body frame.
    q_roll = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    frame = body_to_sensor_frame(0.0, np.zeros(3), q_roll, np.zeros(3), np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(frame.accel_body, [0.0, -9.80665, 0.0], atol=1e-14)


def test_real_udp_reply_peer_timeout_and_cleanup():
    with (
        ArduPilotUDP(port=0, timeout=0.1) as endpoint,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
    ):
        sender.bind(("127.0.0.1", 0))
        sender.settimeout(1.0)
        with pytest.raises(RuntimeError):
            endpoint.send(b"\n{}\n")
        sender.sendto(b"malformed", endpoint.address)
        with pytest.raises(ValueError):
            endpoint.receive()
        assert endpoint.peer is None
        packet = ServoPacket(50, 0, (1500,) * 16)
        sender.sendto(packet.encode(), endpoint.address)
        assert endpoint.receive() == packet
        endpoint.send(stationary_frame().to_json())
        assert json.loads(sender.recvfrom(65535)[0])["timestamp"] == 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as foreign:
            foreign.sendto(packet.encode(), endpoint.address)
            with pytest.raises(ValueError):
                endpoint.receive()
        endpoint.release_peer()
        assert endpoint.peer is None
        with pytest.raises(TimeoutError):
            endpoint.receive()
    with pytest.raises(OSError):
        endpoint.receive()
