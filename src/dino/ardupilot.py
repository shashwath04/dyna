"""ArduPilot JSON external-physics UDP protocol (not MAVLink telemetry).

Binary little-endian servo inputs; newline-delimited JSON physics outputs.
Default listener is loopback:9002. One endpoint represents one SITL instance.
"""

import json
import math
import socket
import struct
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import numpy as np

from ._rotation import rotation
from ._validation import array, unit
from .types import Array


@dataclass(frozen=True)
class ServoPacket:
    """ArduPilot output: rate [Hz], unsigned 32-bit frame counter, raw PWM [us].

    Supports 16 channels (magic 18458) and 32 channels (magic 29569).
    Channels are indexed from zero in Python; PWM values are unsigned 16-bit.
    """

    frame_rate: int
    frame_count: int
    pwm: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.frame_rate) is not int or not 1 <= self.frame_rate <= 65535:
            raise ValueError("frame_rate must be a positive uint16")
        if type(self.frame_count) is not int or not 0 <= self.frame_count <= 0xFFFFFFFF:
            raise ValueError("frame_count must be uint32")
        if len(self.pwm) not in (16, 32):
            raise ValueError("expected 16 or 32 PWM channels")
        if any(type(v) is not int or not 0 <= v <= 65535 for v in self.pwm):
            raise ValueError("PWM channels must be uint16 integers")
        object.__setattr__(self, "pwm", tuple(self.pwm))

    @classmethod
    def decode(cls, payload: bytes) -> "ServoPacket":
        """Reject unknown magic, truncated/extra bytes, or zero frame rate."""
        if len(payload) < 2:
            raise ValueError("truncated servo packet")
        magic = struct.unpack_from("<H", payload)[0]
        channels = {18458: 16, 29569: 32}.get(magic)
        if channels is None or len(payload) != 8 + 2 * channels:
            raise ValueError("invalid servo packet magic/length")
        values = struct.unpack(f"<HHI{channels}H", payload)
        return cls(values[1], values[2], tuple(values[3:]))

    def encode(self) -> bytes:
        """Encode a protocol packet, useful for replay and test benches."""
        magic = 18458 if len(self.pwm) == 16 else 29569
        return struct.pack(
            f"<HHI{len(self.pwm)}H", magic, self.frame_rate, self.frame_count, *self.pwm
        )


@dataclass(frozen=True)
class SensorFrame:
    """Truth/IMU response for SITL. Body is forward-right-down (FRD).

    timestamp is physics time [s], position/velocity are local NED [m, m/s],
    quaternion is Hamilton wxyz body-to-NED, gyro_body is angular rate [rad/s],
    accel_body is specific force [m/s²], NOT body-velocity derivative.
    A stationary level craft reports accel_body=[0,0,-g].
    ArduPilot uses these values to drive its simulated sensors and EKF.
    """

    timestamp: float
    position_ned: Array
    velocity_ned: Array
    quaternion: Array
    gyro_body: Array
    accel_body: Array

    def to_json(self) -> bytes:
        """Validate/copy values and produce one newline-bracketed UTF-8 response.

        Keep SITL time synchronization and lockstep enabled; do not select a
        truth-only estimator. EKF3 selection remains an ArduPilot parameter.
        """
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        payload = {
            "timestamp": self.timestamp,
            "imu": {
                "gyro": array(self.gyro_body, "gyro", (3,)).tolist(),
                "accel_body": array(self.accel_body, "accel_body", (3,)).tolist(),
            },
            "position": array(self.position_ned, "position", (3,)).tolist(),
            "velocity": array(self.velocity_ned, "velocity", (3,)).tolist(),
            "quaternion": unit(self.quaternion, 4, "quaternion").tolist(),
            "no_time_sync": False,
            "no_lockstep": False,
        }
        return ("\n" + json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()


def body_to_sensor_frame(
    timestamp: float,
    position_ned: Array,
    quaternion: Array,
    velocity_body: Array,
    omega_body: Array,
    velocity_dot_body: Array,
    gravity: float = 9.80665,
) -> SensorFrame:
    """Convert rigid-body truth at the IMU/body origin to NED and specific force.

    R maps FRD body to NED; v_N=R v_B;
    f_B=dot(v_B)+omega_B cross v_B-R.T@[0,0,g].
    velocity_body is ABSOLUTE ground-relative body velocity, not water-relative.
    For a displaced IMU, include lever-arm acceleration in a custom observer.
    Caller supplies gyro bias/noise only if deliberately modeling it here.
    """
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity must be finite and positive")
    q = unit(quaternion, 4, "quaternion")
    velocity = array(velocity_body, "velocity_body", (3,))
    omega = array(omega_body, "omega_body", (3,))
    derivative = array(velocity_dot_body, "velocity_dot_body", (3,))
    transform = rotation(q)
    frame = SensorFrame(
        timestamp,
        array(position_ned, "position", (3,)),
        transform @ velocity,
        q,
        omega,
        derivative + np.cross(omega, velocity) - transform.T @ np.array([0.0, 0.0, gravity]),
    )
    frame.to_json()
    return frame


@dataclass
class ArduPilotUDP:
    """Context-managed UDP endpoint; replies go to the actual SITL sender.

    Locks to the first valid sender. Use release_peer() to accept a new source
    port after restarting SITL. Malformed/foreign packets raise ValueError;
    inactivity raises TimeoutError. No background threads or MAVLink connection.
    """

    host: str = "127.0.0.1"
    port: int = 9002
    timeout: float = 2.0
    _socket: socket.socket = field(init=False, repr=False)
    _peer: tuple[str, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 0 <= self.port <= 65535:
            raise ValueError("port must be 0..65535 (0 selects an ephemeral port)")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.settimeout(self.timeout)
            self._socket.bind((self.host, self.port))
        except BaseException:
            self._socket.close()
            raise

    @property
    def address(self) -> tuple[str, int]:
        """Bound IPv4 address and actual port."""
        host, port = self._socket.getsockname()
        return str(host), int(port)

    @property
    def peer(self) -> tuple[str, int] | None:
        return self._peer

    def release_peer(self) -> None:
        """Forget sender identity; caller must also reset the physics runner."""
        self._peer = None

    def receive(self) -> ServoPacket:
        """Receive exactly one datagram; no state change on invalid input."""
        payload, source = self._socket.recvfrom(65535)
        peer = (str(source[0]), int(source[1]))
        if self._peer is not None and peer != self._peer:
            raise ValueError("packet came from a different SITL instance")
        packet = ServoPacket.decode(payload)
        self._peer = peer
        return packet

    def send(self, payload: bytes) -> None:
        """Reply to the last accepted peer. Pass SensorFrame.to_json() output."""
        if self._peer is None:
            raise RuntimeError("receive a valid actuator packet before replying")
        sent = self._socket.sendto(payload, self._peer)
        if sent != len(payload):
            raise OSError("incomplete UDP response")

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
