"""Illustrative 3-DOF Fossen-form boat over ArduRover's JSON UDP interface.

This is a protocol/demo vessel, not an identified model of your boat. Replace
coefficients, thrust curves, geometry and environment with validated vessel data.
See docs/ardupilot_sitl.md for the matching ArduRover setup.
"""

import argparse
from pathlib import Path

import numpy as np

from dino import (
    ArduPilotUDP,
    Array,
    FossenDynamics,
    FossenTerms,
    SensorFrame,
    ServoPacket,
    SITLRunner,
    body_to_sensor_frame,
    rk4_step,
)


def boat_terms(t: float, eta: Array, nu: Array, command: Array) -> FossenTerms:
    """Planar x=[north,east,yaw,surge,sway,yaw_rate], command=[X,Y,N]."""
    surge, sway, yaw_rate = nu
    yaw = eta[2]
    inertia = np.diag([50.0, 60.0, 15.0])
    coriolis = np.array(
        [[0.0, 0.0, -60 * sway], [0.0, 0.0, 50 * surge], [60 * sway, -50 * surge, 0.0]]
    )
    eta_dot = np.array(
        [
            np.cos(yaw) * surge - np.sin(yaw) * sway,
            np.sin(yaw) * surge + np.cos(yaw) * sway,
            yaw_rate,
        ]
    )
    return FossenTerms(inertia, coriolis, np.diag([10.0, 15.0, 5.0]), np.zeros(3), command, eta_dot)


def make_runner(
    left_channel: int = 1, right_channel: int = 3, max_step: float = 0.005
) -> SITLRunner:
    """Use ONE-based servo channel numbers at this example's CLI boundary."""
    if not 1 <= left_channel <= 32 or not 1 <= right_channel <= 32 or left_channel == right_channel:
        raise ValueError("choose distinct servo channels from 1..32")
    dynamics = FossenDynamics(boat_terms)

    def actuators(packet: ServoPacket) -> Array:
        if max(left_channel, right_channel) > len(packet.pwm):
            raise ValueError("selected channel is not present in packet")

        def thrust(channel: int) -> float:
            pwm = packet.pwm[channel - 1]
            # Unconfigured/disabled channels can be zero. Neutral reversible ESC=1500us.
            if pwm == 0:
                return 0.0
            return 40.0 * float(np.clip((pwm - 1500) / 500.0, -1.0, 1.0))

        left, right = thrust(left_channel), thrust(right_channel)
        # Thrusters at y=-0.3m (left), +0.3m (right), both push body-forward.
        return np.array([left + right, 0.0, 0.3 * (left - right)])

    def observe(t: float, x: Array, command: Array) -> SensorFrame:
        yaw = x[2]
        derivative = dynamics(t, x, command)
        return body_to_sensor_frame(
            t,
            np.array([x[0], x[1], 0.0]),
            np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]),
            np.array([x[3], x[4], 0.0]),
            np.array([0.0, 0.0, x[5]]),
            np.array([derivative[3], derivative[4], 0.0]),
        )

    return SITLRunner(
        dynamics, np.zeros(6), actuators, observe, max_step=max_step, stepper=rk4_step
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--max-step", type=float, default=0.005)
    parser.add_argument("--left-channel", type=int, default=1)
    parser.add_argument("--right-channel", type=int, default=3)
    parser.add_argument("--csv", type=Path, default=Path("outputs/ardurover_boat.csv"))
    args = parser.parse_args()
    runner = make_runner(args.left_channel, args.right_channel, args.max_step)
    try:
        with ArduPilotUDP(args.host, args.port, args.timeout) as endpoint:
            print(f"Listening for ArduRover JSON servo packets on {endpoint.address}", flush=True)
            runner.run(endpoint, max_frames=args.frames)
    except KeyboardInterrupt:
        pass
    finally:
        if runner.time > 0:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            runner.result().to_csv(args.csv)
            print(f"Physics time {runner.time:.3f}s; log saved to {args.csv}")


if __name__ == "__main__":
    main()
