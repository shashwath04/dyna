# Dino

An undergraduate GNC simulation and verification project. The documented APIs now
have working implementations, following the student's explicit request to focus on
test benches and pipeline development. Includes forward Euler simulation, sampled
PID, linearization, continuous LQR, quaternion feedback, KF/EKF/MEKF, and a
bit-accurate fixed-point PID/FPGA transaction reference model.

The hardware module is a Python scoreboard model; RTL synthesis and device timing
verification belong to the downstream test bench.

* First Version designed as a baseline w/ the help of Codex. Further versions integrating marine vehicle hydrodynamics and spacecraft dynamics as well as further SITL testing capabilities are currently under development by myself.

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest tests/infrastructure
pytest tests/algorithms -rxX
ruff check .
ruff format --check .
mypy
python -m build
MPLBACKEND=Agg python examples/log_signals.py
```

Import with `import dino` (distribution name: `dino-gnc`). The example exports
supplied samples to `outputs/signals.csv` and `outputs/signals.png`.

See [implementation guide](IMPLEMENTATION_GUIDE.md), [conventions](docs/conventions.md),
and [architecture](docs/architecture.md). Infrastructure tests must pass.
All algorithm acceptance tests run normally: no expected failures or skipped
algorithm bodies. Test benches can import the public functions directly from
`dino`. SciPy supplies ordered Schur decomposition for LQR and matrix exponential
for MEKF propagation; the controller/filter logic resides in Dino.

Run `MPLBACKEND=Agg python examples/closed_loop.py` for a complete scalar closed-loop
example that exports state/command CSV and a plot to `outputs/`.

CI configuration is provided for GitHub; it has not been run remotely.

## ArduRover external physics and nonlinear marine models

Use `ArduPilotUDP` and `SITLRunner` to connect your own plant to ArduRover SITL
through its JSON UDP backend. ArduPilot retains its actual controllers and EKF3.
`FossenDynamics` accepts vessel-supplied inertia, Coriolis, damping, restoring,
force and kinematic terms; `MassMatrixDynamics` accepts arbitrary nonsingular
mass-matrix ODEs. Custom state layouts/actuator maps/observers/steppers are supported.
`rk4_step` is available alongside the existing Euler step.

See [SITL setup and model guide](docs/ardupilot_sitl.md). The runnable
`examples/ardurover_boat.py` includes an illustrative planar twin-thruster model
and a matching parameter file. Replace its example coefficients with your vessel's
validated data. The adapter does not load Simulink files or infer hydrodynamics.
