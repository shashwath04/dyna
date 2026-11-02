"""Bounded live JSON handshake against an installed ArduRover SITL binary.

Runs disarmed with a stationary physics fixture, private temporary files, dynamic
ports and serial telemetry disabled. Does not test mission control or vessel fidelity.
Usage: python scripts/smoke_ardurover.py /absolute/path/to/ardurover
"""

import argparse
import json
import runpy
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from dino import ArduPilotUDP, SensorFrame, SITLRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--boat", action="store_true", help="use the illustrative Fossen boat instead"
    )
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    runner = SITLRunner(
        lambda t, x, u: np.zeros_like(x),
        np.zeros(1),
        lambda packet: np.zeros(1),
        lambda t, x, u: SensorFrame(
            t,
            np.zeros(3),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            np.array([0.0, 0.0, -9.80665]),
        ),
    )
    if args.boat:
        namespace = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "examples/ardurover_boat.py")
        )
        runner = namespace["make_runner"]()
    with tempfile.TemporaryDirectory(prefix="dino-sitl-") as directory:
        root = Path(directory)
        defaults = root / "defaults.parm"
        defaults.write_text("AHRS_EKF_TYPE 3\nEK3_ENABLE 1\nSIM_RATE_HZ 100\n")
        with ArduPilotUDP(port=0, timeout=20.0) as endpoint, (root / "sitl.log").open("w+") as log:
            command = [
                str(binary),
                "--model",
                "JSON:127.0.0.1",
                "--home",
                "41,-87,0,0",
                "--sim-port-out",
                str(endpoint.address[1]),
                "--serial0",
                "none",
                "--serial1",
                "none",
                "--serial2",
                "none",
                "--defaults",
                str(defaults),
                "--rc-in-port",
                "0",
                "--speedup",
                "5",
            ]
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            try:
                if args.frames <= 0:
                    raise ValueError("frames must be positive")
                last = None
                for _ in range(args.frames):
                    last = runner.serve_once(endpoint)
                result = runner.result()
                print(
                    json.dumps(
                        {
                            "frames": result.time.size,
                            "physics_time": runner.time,
                            "last_counter": last.frame_count if last is not None else None,
                            "peer": endpoint.peer,
                        }
                    )
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                log.seek(0)
                print(log.read()[-4000:])


if __name__ == "__main__":
    main()
