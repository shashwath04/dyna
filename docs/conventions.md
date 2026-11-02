# Mathematical conventions

All times are seconds and SI units apply unless a signal explicitly specifies
another unit. NumPy float64 state vectors have shape `(n,)`, controls `(m,)`,
measurements `(p,)`; matrices never use implicit column-vector broadcasting.
State components may have mixed units; Jacobians and covariance entries inherit
pairwise component units. Inputs are never mutated by algorithm APIs.

Quaternions are Hamilton scalar-first `[w,x,y,z]`, active body-to-inertial
rotations. Angular velocity is body-frame radians/s. MEKF errors are right
multiplicative: `q_true = q_nominal ⊗ delta_q`; six error coordinates are body
angle then gyro bias. Normalization, sign equivalence and 180-degree tie-breaking
must be tested explicitly. MEKF process noise is a discrete covariance; deriving
continuous-to-discrete noise is the caller's responsibility.

Simulation starts at zero. Controller samples at zero and integer multiples of
its period, holds between samples, and samples the state before computing the new
command. Integrator splits at all events and final time. Log after each event and
integration endpoint; record state and held command under `state` and `command`.
Integration uses forward Euler with a maximum step length restarted after each
endpoint. Controller events are computed from an integer sample index, not repeated
addition. A control event within four floating-point spacings of the final time is
snapped to that time; a coincident final event runs the controller and is logged
once. Zero duration yields one initial sample. Unrepresentable positive steps
raise ValueError instead of hanging.

Validation rejects nonfinite and complex data and wrong shapes. Covariance
symmetry and PSD tolerances are 1e-10 relative to the spectral norm. Roundoff-sized
negative PSD eigenvalues are projected to zero; material negatives are rejected.
R must be strictly positive definite. Unit quaternions and direction vectors must
have norm within 1e-8 of one; accepted roundoff is normalized.

MEKF uses constant bias-corrected body rate per sample, exact nominal exponential
propagation, F=[-[omega]x,-I;0,0], and exp(F dt). The vector observation Jacobian is
H=[[predicted_body]x,0]. Correction is injected with a quaternion exponential and
covariance reset with the SO(3) right Jacobian. One vector does not observe rotation
about itself. The MEKF remains a local filter requiring a reasonable initial attitude.

Fixed-point arithmetic uses Python integers, widened differences/products/sums,
and separately rounded products (nearest, ties-to-even). Output and integral
saturate only at their documented boundaries. Raw arguments and widths must be
Python int (bool is rejected); convert NumPy scalar integers at the boundary.
A transaction call represents one rising edge: valid input produces valid output
in returned registered state; idle holds registers and deasserts valid; reset
clears registers and valid, overriding input. Configuration is always validated;
reset ignores old state/input and idle ignores input. No pipeline or backpressure
is implied beyond this single register transition.
