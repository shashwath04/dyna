# Dino Versions
## Dino 0.1.0.dev0
---

Provides a working Python library for simulation, controller design, state estimation, experiment orchestration, and hardware reference calculations. All previously stubbed algorithm functions are implemented.

#### Simulation
| API | Capability
|:---:|:---:|
|`integrate_step`| Advances a user-supplied dynamics model using forward Euler. Validates inputs and derivative shapes, preserves caller arrays
|`simulate`|Runs a continuous plant with a periodically sampled controller. Holds commands between updates, splits integration at control events, handles final time, and logs state and command.|

Current limitations: forward Euler only, one controller sampling period; fixed array reference. There is no built-in RK4 or any higher order intergration schema. There is no adaptive integration, or seperate sensor/estimation schema

#### Control
|API|Capabilities|
|:---:|:---:|
|`kalman_predict`|Linear discrete state and covariance prediction|
|`kalman_update`|Linear measurement correction using an innovation to solve and Joseph-form covariance update|
|`ekf_predict`|Nonlinear state prediction using your transition function and supplied Jacobian.|
|`ekf_update`|Nonlinear measurement correction using your observation function and supplied Jacobian.|
|`mekf_step`|Quaternion attitude and gyro-bias estimation using gyro data and one measured direction vector. Includes propagation, correction, quaternion injection, and covariance reset.|

Conventions are Hamilton scalar-first quaternions `[w, x, y, z]`, body-to-inertial orientation, and body-frame angular velocity.

Current limits are that the EKF Jacobians and discrete noise covariance come from user. MEKF assumes a reasonable initial attitude and constant measured rate over each step. One direction vector cannot observe rotation about itself. - Estimator calls must be connected to your simulation or test-bench pipeline explicitly.

#### Fixed-Point and FPGA Verification

|API|Capabilities|
|:---:|:---:|
|`fixed_point_pid_step`|Signed fixed-point PID with exact integer arithmetic, separately rounded products, ties-to-even rounding, output saturation, and integral-register saturation.|
|`fpga_controller_step`| Models one synchronous register transition, including valid input, idle hold, and reset priority.|
|`FixedPIDConfig`, `FPGAState`| Explicit coefficient, format, and register-state containers for reproducible comparisons. |

These functions can supply expected outputs for an RTL scoreboard. Fixed-point PID currently omits anti-windup.

The hardware layer models the documented arithmetic and register behavior. RTL generation, synthesis, device interfaces, configurable pipeline latency, and timing verification remain outside this version.

## ArduRover SITL extension (0.1.0.dev0)

Added ArduPilot JSON UDP physics transport, 16/32-channel codec, specific-force
conversion, lockstep custom-state runner, mass-matrix/Fossen dynamics adapters,
RK4 stepping, planar boat demo and a bounded live ArduRover handshake utility.
Existing offline APIs are retained. Vessel coefficients and actuation are supplied
by the application. See docs/ardupilot_sitl.md for setup and validation scope.
