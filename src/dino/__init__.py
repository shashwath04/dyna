"""Dino simulation, control, estimation, and verification infrastructure."""

from .ardupilot import ArduPilotUDP, SensorFrame, ServoPacket, body_to_sensor_frame
from .control import attitude_control, linearize, lqr, pid_step
from .estimation import ekf_predict, ekf_update, kalman_predict, kalman_update, mekf_step
from .experiments import Trial, monte_carlo, parameter_sweep
from .hardware import FixedPIDConfig, FPGAState, fixed_point_pid_step, fpga_controller_step
from .logging import SignalLogger, SignalSpec, SimulationResult
from .marine import FossenDynamics, FossenModel, FossenTerms, MassMatrixDynamics, MassMatrixTerms
from .simulation import integrate_step, rk4_step, simulate
from .sitl import ActuatorMap, FrameSequenceError, PhysicsStepper, SensorObserver, SITLRunner
from .types import Array, AttitudeEstimate, Controller, Dynamics, Estimate, PIDState, StateMap
from .visualization import plot_signal

__all__ = [
    "ActuatorMap",
    "ArduPilotUDP",
    "Array",
    "AttitudeEstimate",
    "Controller",
    "Dynamics",
    "Estimate",
    "FPGAState",
    "FixedPIDConfig",
    "FossenDynamics",
    "FossenModel",
    "FossenTerms",
    "FrameSequenceError",
    "MassMatrixDynamics",
    "MassMatrixTerms",
    "PIDState",
    "PhysicsStepper",
    "SITLRunner",
    "SensorFrame",
    "SensorObserver",
    "ServoPacket",
    "SignalLogger",
    "SignalSpec",
    "SimulationResult",
    "StateMap",
    "Trial",
    "attitude_control",
    "body_to_sensor_frame",
    "ekf_predict",
    "ekf_update",
    "fixed_point_pid_step",
    "fpga_controller_step",
    "integrate_step",
    "kalman_predict",
    "kalman_update",
    "linearize",
    "lqr",
    "mekf_step",
    "monte_carlo",
    "parameter_sweep",
    "pid_step",
    "plot_signal",
    "rk4_step",
    "simulate",
]
