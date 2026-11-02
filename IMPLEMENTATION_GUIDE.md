# Student implementation guide

All documented algorithm APIs are implemented following your explicit request to
focus on test benches and pipeline work. This guide retains their mathematical
prerequisites, derivation exercises, dependencies and acceptance targets for review.
There are no remaining algorithm stubs or expected-failure acceptance markers.
The original student-owned boundary still applies to future work unless explicitly
expanded by you.

Implementation choices: forward Euler integration; central-difference linearization;
back-calculation PID; Hamiltonian stable-subspace continuous LQR; Joseph-form KF/EKF;
right-error MEKF with exponential injection/reset; integer ties-to-even fixed PID.
See [conventions](docs/conventions.md) for timing and numerical tolerances.

## Suggested verification sequence and dependencies

1. Derive conventions and state units; implement `integrate_step`, then `simulate`.
2. Implement `pid_step`, `linearize`, and then `lqr` for a chosen plant.
3. Derive quaternion kinematics and implement `attitude_control`.
4. Implement Kalman prediction/update, EKF prediction/update, then `mekf_step`.
5. Derive a fixed-point range budget, implement `fixed_point_pid_step` then
   `fpga_controller_step`; compare RTL transactions to the software contract.

Dependencies below identify conceptual prerequisites; infrastructure imports do not
execute any of them. Acceptance baseline is in `tests/algorithms/test_acceptance.py`.
Each listed additional case is work to complete before claiming full acceptance.
Tests use analytic reference values rather than calling a second copy of the algorithm.
Additional coverage is in `test_control.py`, `test_estimation.py`, `test_hardware.py`
and `test_simulation.py` in the same directory.

## `integrate_step`

- Prerequisites: ODEs, local/global truncation error; Taylor expansion.
- Derivation task: Choose a method and derive its order before coding stages.
- Acceptance: constant derivative test; add exponential decay convergence, dt validation, input preservation.
- Dependencies: None.
- Theory: Hairer, Nørsett & Wanner, Solving Ordinary Differential Equations I.
- Equations and contract: `src/dino/simulation.py`, docstring for this function.

## `simulate`

- Prerequisites: sample/hold systems, event ordering, floating-point time.
- Derivation task: Draw a timeline with incommensurate integration and control periods.
- Acceptance: event/hold test; add zero duration, coincident final event, noninteger periods and invalid periods.
- Dependencies: integrate_step.
- Theory: Franklin, Powell & Workman, Digital Control of Dynamic Systems.
- Equations and contract: `src/dino/simulation.py`, docstring for this function.

## `pid_step`

- Prerequisites: discrete derivatives/integrals, saturation, tracking feedback.
- Derivation task: Track command and integrator units; explain back-calculation feedback sign.
- Acceptance: hand-calculated saturation test; add unsaturated sequences, recovery, bounds and derivative history.
- Dependencies: None.
- Theory: Åström & Murray, Feedback Systems; Åström & Hägglund, Advanced PID Control.
- Equations and contract: `src/dino/control.py`, docstring for this function.

## `linearize`

- Prerequisites: Jacobians and finite differences.
- Derivation task: Derive analytic derivatives of a small nonlinear model, then compare perturbations.
- Acceptance: polynomial Jacobian test; add multi-state cross terms, epsilon sensitivity and shape errors.
- Dependencies: None.
- Theory: Åström & Murray, Feedback Systems.
- Equations and contract: `src/dino/control.py`, docstring for this function.

## `lqr`

- Prerequisites: quadratic forms, stabilizability, detectability, Riccati equation.
- Derivation task: Derive scalar CARE and distinguish stabilizing from non-stabilizing roots.
- Acceptance: analytic scalar gain test; add CARE residual, closed-loop eigenvalues, singular R and uncontrollable unstable mode.
- Dependencies: linearize for nonlinear plant use.
- Theory: Tedrake, Underactuated Robotics, LQR chapter.
- Equations and contract: `src/dino/control.py`, docstring for this function.

## `attitude_control`

- Prerequisites: Hamilton products, frames, SO(3), quaternion double cover.
- Derivation task: Derive torque direction for a single-axis small error before three-axis tests.
- Acceptance: sign-equivalence torque test; add identity, damping-only, 180-degree tie and norm rejection.
- Dependencies: None.
- Theory: Wie, Space Vehicle Dynamics and Control; Shuster, A Survey of Attitude Representations.
- Equations and contract: `src/dino/control.py`, docstring for this function.

## `kalman_predict`

- Prerequisites: Gaussian propagation, covariance units, PSD matrices.
- Derivation task: Propagate a scalar random variable by hand.
- Acceptance: scalar mean/variance test; add coupled covariance, PSD and nonmutation checks.
- Dependencies: None.
- Theory: Särkkä & Svensson, Bayesian Filtering and Smoothing.
- Equations and contract: `src/dino/estimation.py`, docstring for this function.

## `kalman_update`

- Prerequisites: Gaussian conditioning, innovation, Joseph form.
- Derivation task: Derive a scalar posterior from the product of two Gaussians.
- Acceptance: equal-variance posterior test; add correlated state, repeated measurements and malformed R.
- Dependencies: kalman_predict for full cycle.
- Theory: Särkkä & Svensson, Bayesian Filtering and Smoothing.
- Equations and contract: `src/dino/estimation.py`, docstring for this function.

## `ekf_predict`

- Prerequisites: first-order uncertainty propagation and Jacobians.
- Derivation task: Explain why f(mean) differs from the exact transformed Gaussian mean.
- Acceptance: nonlinear square map test; add linear equivalence, invalid Jacobian and PSD checks.
- Dependencies: kalman_predict concepts.
- Theory: Särkkä & Svensson, Bayesian Filtering and Smoothing.
- Equations and contract: `src/dino/estimation.py`, docstring for this function.

## `ekf_update`

- Prerequisites: nonlinear observations, local linearization.
- Derivation task: Derive observation Jacobian at the prior rather than updated mean.
- Acceptance: nonlinear scalar posterior test; add linear equivalence, zero innovation and bad dimensions.
- Dependencies: kalman_update concepts.
- Theory: Särkkä & Svensson, Bayesian Filtering and Smoothing.
- Equations and contract: `src/dino/estimation.py`, docstring for this function.

## `mekf_step`

- Prerequisites: rotation kinematics, error-state dynamics, observability, covariance reset.
- Derivation task: Derive right-error transition and vector observation Jacobians with declared frames.
- Acceptance: stationary/zero innovation test; add analytic single-axis propagation, sign equivalence, bias correction, reset covariance and one-vector nullspace.
- Dependencies: EKF concepts and quaternion conventions.
- Theory: Markley, Attitude Error Representations for Kalman Filtering; Crassidis & Junkins, Optimal Estimation of Dynamic Systems.
- Equations and contract: `src/dino/estimation.py`, docstring for this function.

## `fixed_point_pid_step`

- Prerequisites: binary scaling, quantization, rounding and overflow.
- Derivation task: Write a range budget for each register and intermediate.
- Acceptance: exact binary PID test; add half-LSB ties, negative products, saturation and register extremes.
- Dependencies: pid_step concepts; fixed contract omits anti-windup.
- Theory: Lyons, Understanding Digital Signal Processing; target FPGA arithmetic documentation.
- Equations and contract: `src/dino/hardware.py`, docstring for this function.

## `fpga_controller_step`

- Prerequisites: synchronous registers, reset, valid latency, bit-accurate verification.
- Derivation task: Draw clock-by-clock transactions before writing software or RTL.
- Acceptance: reset/hold/valid test; add valid bursts and scoreboard comparison to fixed-point model.
- Dependencies: fixed_point_pid_step.
- Theory: Harris & Harris, Digital Design and Computer Architecture; target FPGA documentation.
- Equations and contract: `src/dino/hardware.py`, docstring for this function.

## Governing equations index

- Plant: dx/dt=f(t,x,u); discrete sample/hold u(t)=u_k for t_k<=t<t_(k+1).
- PID: v_k=kp e_k+I_k+kd(e_k-e_(k-1))/dt; u_k=sat(v_k);
  I_(k+1)=I_k+dt[ki e_k+k_aw(u_k-v_k)].
- Linearization: delta_x_dot=A delta_x+B delta_u, A=df/dx, B=df/du.
- LQR: J=integral(xᵀQx+uᵀRu)dt; AᵀP+PA-PBR⁻¹BᵀP+Q=0;
  K=R⁻¹BᵀP and u=-Kx.
- Quaternion torque: tau=-2 kp vec(q_error)-kd omega, with shortest-arc sign.
- KF: x⁻=Fx+Bu, P⁻=FPFᵀ+Q; S=HP⁻Hᵀ+R; K=P⁻HᵀS⁻¹;
  x⁺=x⁻+K(z-Hx⁻), P⁺=(I-KH)P⁻(I-KH)ᵀ+KRKᵀ.
- EKF replaces linear means by f(x), h(x), with Jacobians at the prior mean.
- MEKF: q_dot=0.5 q⊗[0,omega-b]; q_true=q_nom⊗delta_q;
  delta_q≈[1,delta_theta/2]. Derive transition, injection and reset Jacobians yourself.
- Fixed-point raw representation: real=raw*2⁻ᶠ; quantized product is
  round_ties_even(a_raw*b_raw/2ᶠ). Widen before arithmetic, saturate at documented boundaries.
- FPGA: register transition follows reset, then valid, then hold priority.

## Reading links and review workflow

[Tedrake's LQR notes](https://underactuated.mit.edu/lqr.html) cover the stabilizing
Riccati solution. [AHRS EKF documentation](https://ahrs.readthedocs.io/en/stable/filters/ekf.html)
provides an additional quaternion filtering reference; reconcile its conventions
with this project's before using equations. These are theory references, not
instructions to copy a library implementation.

Before coding each algorithm, record your derivation and dimensional checks in a
personal design note. Submit it with your implementation for review. Debug from
small independent reference cases; request hints first. A full implementation
requires an explicit request from you. Passing the starter tests alone does not
establish numerical robustness or hardware correctness.

## Implementation references

- SciPy matrix routines: https://docs.scipy.org/doc/scipy/reference/linalg.html
  (ordered Schur decomposition and matrix exponential).
- Markley, Attitude Error Representations for Kalman Filtering:
  https://ntrs.nasa.gov/citations/20020060647 . Reconcile quaternion ordering and
  error-side conventions with this project's declared Hamilton/right-error form.

## ArduRover and marine model extension

Implemented UDP transport, lockstep physics runner, body/NED/IMU conversion,
RK4 stepping and mass-matrix/Fossen adapters are described in
[the SITL guide](docs/ardupilot_sitl.md). The new model callback interfaces are
intentional Protocol contracts, not unfinished algorithm bodies. Vessel-specific
hydrodynamics, actuator maps and sensor offsets must come from the boat project.
