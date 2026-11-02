"""Lockstep external-physics runner independent of controller/estimator algorithms."""

import math
from collections.abc import Callable
from typing import Protocol

from ._validation import array, vector
from .ardupilot import ArduPilotUDP, SensorFrame, ServoPacket
from .logging import SignalLogger, SignalSpec, SimulationResult
from .simulation import integrate_step
from .types import Array, Dynamics


class ActuatorMap(Protocol):
    """Map raw servo channels to a model-defined command vector (forces or inputs).

    Read ServoPacket.pwm using zero-based indexes. No motor layout or PWM range
    is assumed. Stateful actuator dynamics should live in the plant state.
    """

    def __call__(self, packet: ServoPacket) -> Array: ...


class SensorObserver(Protocol):
    """Map arbitrary plant state and held command to a timestamped SITL response.

    Pose/velocity/IMU conversions belong here, not in the transport. The callback
    receives copies and must return the supplied physics timestamp unchanged.
    """

    def __call__(self, t: float, x: Array, u: Array) -> SensorFrame: ...


class PhysicsStepper(Protocol):
    """Advance a plant over dt seconds; return same-shaped state without mutation.

    Implement this interface for RK4/adaptive solvers, quaternion normalization,
    external engines, constrained states, or model-specific integration.
    """

    def __call__(self, dynamics: Dynamics, t: float, x: Array, u: Array, dt: float) -> Array: ...


class FrameSequenceError(ValueError):
    """Missing/out-of-order frames or inconsistent retransmissions require recovery."""


class SITLRunner:
    """Advance custom dynamics once per ArduPilot actuator frame.

    Each new frame advances 1/frame_rate seconds, including the first frame.
    Substeps are <= max_step; held actuator input is mapped once per frame.
    Physics time is independent of wall time. Duplicate frames resend cached
    bytes without advancing the plant, RNG, observer, or actuator callbacks.

    Frame zero after a nonterminal counter resets to x0/time zero (new episode).
    uint32 counter rollover is accepted. Other gaps/backward frames raise
    FrameSequenceError rather than silently guessing missing actuator history.
    A delayed old frame zero is indistinguishable from restart; use one ordered
    local SITL session per runner. If the sender port changes, release_peer()
    and reset() explicitly. Callbacks should be deterministic/pure; optional
    on_reset resets caller-owned RNGs or stateful closures.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        x0: Array,
        actuators: ActuatorMap,
        observe: SensorObserver,
        *,
        max_step: float = 0.01,
        stepper: PhysicsStepper = integrate_step,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        if not math.isfinite(max_step) or max_step <= 0:
            raise ValueError("max_step must be finite and positive")
        self.dynamics = dynamics
        self.actuators = actuators
        self.observe = observe
        self.max_step = max_step
        self.stepper = stepper
        self.on_reset = on_reset
        self._initial = vector(x0, "x0")
        self._state = self._initial.copy()
        self.time = 0.0
        self._last: ServoPacket | None = None
        self._response: bytes | None = None
        self._logger: SignalLogger | None = None
        self.restarts = 0

    @property
    def state(self) -> Array:
        """Independent current-state snapshot."""
        return self._state.copy()

    def reset(self) -> None:
        """Start a fresh episode and discard that episode's cached reply/log.

        Export result() before manual reset if the previous history is needed.
        Restart detection also clears the log; restarts counts automatic resets.
        """
        if self.on_reset is not None:
            self.on_reset()
        self._state = self._initial.copy()
        self.time = 0.0
        self._last = None
        self._response = None
        self._logger = None

    def result(self) -> SimulationResult:
        """Response-endpoint state/command history for the current episode."""
        if self._logger is None:
            raise RuntimeError("no completed physics frame yet")
        return self._logger.result()

    def handle(self, packet: ServoPacket) -> bytes:
        """Process a decoded packet; returns JSON bytes ready for UDP reply."""
        if self._last is not None:
            delta = (packet.frame_count - self._last.frame_count) & 0xFFFFFFFF
            if delta == 0:
                if packet != self._last:
                    raise FrameSequenceError("same frame counter with changed payload")
                assert self._response is not None
                return self._response
            if delta != 1:
                if packet.frame_count == 0:
                    self.reset()
                    self.restarts += 1
                else:
                    raise FrameSequenceError(
                        "missing/out-of-order frame; explicitly reset the session"
                    )
        command = vector(self.actuators(packet), "actuator command")
        if self._logger is not None:
            expected = self._logger.specs["command"].width
            if command.shape != (expected,):
                raise ValueError("actuator command shape must remain constant")
        target = self.time + 1.0 / packet.frame_rate
        if not math.isfinite(target) or target <= self.time:
            raise ValueError("physics time cannot advance at this resolution")
        state, time = self._state.copy(), self.time
        while time < target:
            stop = min(time + self.max_step, target)
            if stop <= time:
                raise ValueError("physics substep cannot advance at this resolution")
            state = array(
                self.stepper(self.dynamics, time, state.copy(), command.copy(), stop - time),
                "integrated state",
                self._initial.shape,
            )
            time = stop
        frame = self.observe(target, state.copy(), command.copy())
        if frame.timestamp != target:
            raise ValueError("observer timestamp must equal physics time")
        response = frame.to_json()
        if self._logger is None:
            self._logger = SignalLogger(
                {
                    "state": SignalSpec(state.size, "model-defined SI"),
                    "command": SignalSpec(command.size, "model-defined SI"),
                }
            )
        self._logger.append(target, {"state": state, "command": command})
        self._state, self.time = state, target
        self._last, self._response = packet, response
        return response

    def serve_once(self, endpoint: ArduPilotUDP) -> ServoPacket:
        """Receive, advance/replay, then reply. Socket timeout errors propagate."""
        packet = endpoint.receive()
        endpoint.send(self.handle(packet))
        return packet

    def run(self, endpoint: ArduPilotUDP, *, max_frames: int | None = None) -> SimulationResult:
        """Serve a bounded number of transactions, or until interrupted if None.

        Retransmissions count toward max_frames but do not add log samples.
        Timeouts/malformed packets/sequence gaps propagate for the supervisor to
        handle; export result() there if partial logs are needed.
        """
        if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer or None")
        count = 0
        while max_frames is None or count < max_frames:
            self.serve_once(endpoint)
            count += 1
        return self.result()
