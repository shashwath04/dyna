# ArduRover SITL with custom Python dynamics

Dino implements ArduPilot's JSON external-physics UDP backend. ArduRover continues
to run its own guidance, controllers and EKF3; Dino provides physical truth/IMU
responses. This socket is NOT MAVLink: use MAVProxy, a GCS or pymavlink separately
for arming, modes, missions, parameters and telemetry.

## Quick start: demo vessel

Install Dino in the Python environment used for the simulator:

```sh
python -m pip install -e /absolute/path/to/Dino
```

From the Dino checkout start the physics backend:

```sh
python examples/ardurover_boat.py --frames 10000
```

In your ArduPilot environment, start Rover using the JSON model. For example, from
its Tools/autotest directory:

```sh
./sim_vehicle.py -v Rover -f rover-skid --model JSON:127.0.0.1 \
  --add-param-file /absolute/path/to/Dino/examples/ardurover_boat.parm \
  --console --map
```

The example binds 127.0.0.1:9002; its startup timeout is 30 wall-clock seconds.
Start the backend near the SITL launch or increase --timeout. Start a fresh SITL
session/working directory when changing models; existing saved parameters may
need explicit verification. The supplied parameter file selects EKF3 and maps
servo 1 to left throttle and servo 3 to right throttle. Verify the actual loaded
parameters in your ArduPilot build. No code auto-arms or uploads a mission.
Use your GCS to complete setup and inspect estimator status before commanding motion.

The demo assumes reversible ESCs: neutral 1500us, +/-40N at 1000/2000us, 0.6m
thruster separation. Zero PWM is treated as disabled. Its mass, damping and
thrust values are illustrative; they are NOT identified parameters of your boat.
Use --left-channel/--right-channel to change one-based servo channel numbers.
Library ServoPacket.pwm itself uses zero-based indexes. --csv sets the output path.
No visualizer or GCS is included in the Python example.

## Replace the dynamics

Any callable dynamics(t,x,u)->x_dot remains supported. State can have any nonempty
flat-vector shape, including appended motor, actuator, battery or environmental
states. No state-space linearity or standard ordering is required.

For M(t,x,u) x_dot = rhs(t,x,u), use MassMatrixDynamics with a callback returning
(M,rhs). M must be nonsingular; this is an ODE adapter, not a DAE solver.

For a Fossen-form model, supply a callback:

```python
from dino import FossenDynamics, FossenTerms

# Your function must compute and return vessel-specific terms:
# def vessel_terms(t, eta, nu, command) -> FossenTerms:
#     return FossenTerms(M, C, D, g, tau, eta_dot)
# dynamics = FossenDynamics(vessel_terms, pose_size=3, velocity_size=3)
```

The adapter assembles

    x = [eta, nu]
    eta_dot = kinematics
    M nu_dot = tau - C nu - D nu - g

It solves the inertia system without explicitly inverting M. The callback owns
added mass, Coriolis/damping laws, hydrostatics, kinematics, current corrections,
wind/waves, thrust curves and parameter units. No coefficients are inferred.
M must be symmetric positive definite. Shape-checking C/D does not prove physical
passivity or correct hydrodynamics; verify these in your model tests.

Use sizes (3,3) for planar north/east/yaw and surge/sway/yaw_rate; (6,6) for Euler
6-DOF; (7,6) for position/quaternion poses. Supply the correct eta_dot for your
layout. Quaternion integration needs a custom PhysicsStepper to preserve unit
norm. Current-relative formulations require consistent complete force terms;
C(nu)nu+D(nu)nu is not a universal substitute for every relative-current equation.
For formulations that do not fit this split, use MassMatrixDynamics or your own
Dynamics callback.

## Connect a model to the runner

SITLRunner takes four main inputs:

1. dynamics: arbitrary state derivative callback.
2. x0: your initial state vector.
3. actuators: ServoPacket -> model command vector; define the actual propulsion map.
4. observe: (physics_time,state,command) -> SensorFrame.

Supply max_step to bound physics substeps; default stepping is forward Euler.
Pass stepper=rk4_step for smooth nonlinear models, or a PhysicsStepper callback
for your preferred solver/normalization. The demo uses RK4. Solver accuracy and
stability still require comparison with Simulink and timestep-convergence tests.

Call runner.serve_once(endpoint) in your supervisor loop or runner.run(endpoint,
max_frames=N). run() returns state/command histories; result().to_csv(path) can
export partial data after a timeout. Logs contain response endpoints, not an
initial t=0 sample. They currently omit raw PWM and ArduPilot estimator outputs;
capture ServoPacket and MAVLink/DataFlash separately for those comparisons.

## Frame conventions and sensors

- NED: north, east, down; local position in metres relative to the simulation origin.
- Body FRD: forward, right, down; absolute ground-relative linear velocity in m/s.
- Quaternion: Hamilton scalar-first wxyz, body-to-NED. Gyro: body rad/s.
- accel_body: accelerometer specific force in body m/s².

body_to_sensor_frame constructs the mandatory frame using:

    velocity_N = R velocity_B
    specific_force_B = velocity_dot_B + omega_B cross velocity_B - R.T [0,0,g]

A stationary level boat must send approximately [0,0,-9.80665] for accel_body.
Do not pass nu_dot directly as the IMU acceleration, or substitute water-relative
velocity for ground-relative velocity. The helper assumes the IMU is at the body
origin. Account for lever arms yourself if needed. A custom observer can generate
SensorFrame directly for a different state ordering or sensor model.

The mandatory response covers position, velocity, quaternion and IMU values.
Optional external fields such as windvane/rangefinder/battery are not exposed by
SensorFrame yet. ArduPilot's SITL drivers synthesize other sensors from truth;
configure ArduPilot noise/bias settings deliberately rather than adding duplicate
noise accidentally. SensorFrame keeps time sync and lockstep enabled; selecting
EKF3 is the firmware configuration's responsibility.

## Transport and time semantics

- ArduPilot sends binary servo packets; Dino replies with newline-bracketed JSON
  to the packet's actual source IP/port. Do not send MAVLink to this listener.
- Both 16-channel magic 18458 and 32-channel magic 29569 are supported. Packet
  layout is little-endian uint16 magic/rate, uint32 counter, then uint16 PWM.
- Each new frame advances physics by 1/frame_rate seconds, including the first.
  Physics substeps may be smaller, with one held command for the entire frame.
  This is lockstep simulation, not wall-clock real-time scheduling.
- Identical repeated frames resend the exact cached response, without stepping,
  remapping actuators, resampling noise or logging twice.
- Rate changes and uint32 counter rollover are supported. Counter zero after an
  established session restarts x0/time/log. Export prior episodes before reset
  if needed; on_reset can reset caller-owned random generators or closures.
- Missing/backward frames and changed duplicate payloads raise FrameSequenceError.
  They do not silently extrapolate unknown commands. Reset/reconnect explicitly.
  An old delayed frame zero cannot be distinguished from a restart.
- The endpoint locks to its first valid sender. A restart using a new source port
  requires endpoint.release_peer() plus runner.reset(). Each SITL instance needs
  its own endpoint/runner/port. Binding outside loopback is an explicit option.
- Socket timeouts, invalid payloads and callback failures propagate. There is no
  background retry thread; the application decides how to recover and save logs.

## Verification and references

The automated suite includes independent 16/32-channel packet fixtures, malformed
input checks, live loopback UDP exchanges, duplicate/reset/rollover/rate handling,
frame transforms/specific force, analytic coupled-inertia equations and RK4
convergence. It does not require an ArduPilot installation.

For a bounded live handshake against your own SITL binary:

```sh
python scripts/smoke_ardurover.py /absolute/path/to/build/sitl/bin/ardurover
```

This launches a separate disarmed process with temporary files and dynamically
selected physics port, exchanges 100 stationary frames, then terminates it. It
checks the wire handshake only, not vessel fidelity, PID tuning or EKF convergence.
Pass --boat to exercise the illustrative Fossen boat through the same connection.

Protocol sources checked against ArduPilot's current repository:
- https://github.com/ArduPilot/ardupilot/blob/master/libraries/SITL/examples/JSON/readme.md
- https://github.com/ArduPilot/ardupilot/blob/master/libraries/SITL/SIM_JSON.h
- https://github.com/ArduPilot/ardupilot/blob/master/libraries/SITL/SIM_JSON.cpp
- Fossen model: https://www.fossen.biz/html/marineCraftModel.html

Keep the Simulink model as a reference: compare identical open-loop PWM/force
inputs, rest equilibrium, surge, turns and coast-down before assessing missions.
