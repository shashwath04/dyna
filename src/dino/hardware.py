"""Bit-accurate fixed-point PID and synchronous transaction reference model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedPIDConfig:
    """Pre-scaled gains, signed Q-format widths, and raw actuator bounds."""

    kp_raw: int
    ki_dt_raw: int
    kd_over_dt_raw: int
    fractional_bits: int
    word_bits: int
    lower_raw: int
    upper_raw: int


@dataclass(frozen=True)
class FPGAState:
    """Registered raw command, integrator and error; valid marks output availability."""

    output_raw: int = 0
    integral_raw: int = 0
    previous_error_raw: int = 0
    valid: bool = False


def fixed_point_pid_step(
    error_raw: int,
    integral_raw: int,
    previous_error_raw: int,
    kp_raw: int,
    ki_dt_raw: int,
    kd_over_dt_raw: int,
    fractional_bits: int,
    word_bits: int,
    lower_raw: int,
    upper_raw: int,
) -> tuple[int, int, int]:
    """Signed Q-format PID without anti-windup; return output, integral_next, error_raw.
    All stored values share scale 2^-fractional_bits; gains preinclude sample period.
    v=kp*e+I+kd_over_dt*(e-e_prev); I_next=I+ki_dt*e.
    Each product rounds separately, nearest ties-to-even after rescaling; widened intermediates,
    saturation only at output bounds and integral register to signed word_bits.
    Reject values outside format, fractional_bits<0 or >=word_bits-1, invalid bounds.
    """
    _validate_format(fractional_bits, word_bits)
    values = (
        error_raw,
        integral_raw,
        previous_error_raw,
        kp_raw,
        ki_dt_raw,
        kd_over_dt_raw,
        lower_raw,
        upper_raw,
    )
    low, high = -(1 << (word_bits - 1)), (1 << (word_bits - 1)) - 1
    if any(type(v) is not int or not low <= v <= high for v in values):
        raise ValueError("raw values must be Python integers within signed word range")
    if lower_raw > upper_raw:
        raise ValueError("reversed actuator bounds")
    # Each product is rounded separately before additions, with unlimited-width
    # Python integer intermediates. Never route raw arithmetic through floats.
    proportional = _round_product(kp_raw * error_raw, fractional_bits)
    derivative = _round_product(kd_over_dt_raw * (error_raw - previous_error_raw), fractional_bits)
    increment = _round_product(ki_dt_raw * error_raw, fractional_bits)
    output = min(upper_raw, max(lower_raw, proportional + integral_raw + derivative))
    integral = min(high, max(low, integral_raw + increment))
    return output, integral, error_raw


def fpga_controller_step(
    error_raw: int,
    state: FPGAState,
    coefficients: FixedPIDConfig,
    valid: bool = True,
    reset: bool = False,
) -> FPGAState:
    """One rising-edge transaction for the fixed-point PID contract; software RTL reference.
    Reset takes precedence and clears output/integral/previous_error, valid=False.
    Valid input advances PID with one-cycle latency represented by returned state.valid.
    Invalid input holds registers and returns valid=False. No backpressure.
    Registers and coefficients use the signed Q format in FixedPIDConfig.
    Compare RTL valid cycles bit-for-bit with fixed_point_pid_step; reject malformed config.
    """
    # Validate configuration even on reset/idle clocks. Input/state registers
    # are only consumed on an active non-reset clock.
    config = coefficients
    fixed_point_pid_step(
        0,
        0,
        0,
        config.kp_raw,
        config.ki_dt_raw,
        config.kd_over_dt_raw,
        config.fractional_bits,
        config.word_bits,
        config.lower_raw,
        config.upper_raw,
    )
    if type(valid) is not bool or type(reset) is not bool:
        raise ValueError("valid and reset must be bool")
    if reset:
        return FPGAState()
    # Validate held state even on idle clocks; output is a register value, which
    # may be zero after reset even when zero lies outside actuator bounds.
    low, high = -(1 << (config.word_bits - 1)), (1 << (config.word_bits - 1)) - 1
    if any(
        type(v) is not int or not low <= v <= high
        for v in (state.output_raw, state.integral_raw, state.previous_error_raw)
    ):
        raise ValueError("state registers must fit signed word range")
    if not valid:
        return FPGAState(state.output_raw, state.integral_raw, state.previous_error_raw, False)
    output, integral, previous = fixed_point_pid_step(
        error_raw,
        state.integral_raw,
        state.previous_error_raw,
        config.kp_raw,
        config.ki_dt_raw,
        config.kd_over_dt_raw,
        config.fractional_bits,
        config.word_bits,
        config.lower_raw,
        config.upper_raw,
    )
    return FPGAState(output, integral, previous, True)


def _validate_format(fractional_bits: int, word_bits: int) -> None:
    """Require an explicit signed format with at least one non-sign integer bit."""
    if type(word_bits) is not int or type(fractional_bits) is not int:
        raise ValueError("format widths must be Python integers")
    if word_bits < 2 or not 0 <= fractional_bits < word_bits - 1:
        raise ValueError("require word_bits>=2 and 0<=fractional_bits<word_bits-1")


def _round_product(product: int, fractional_bits: int) -> int:
    """Exact division by 2**f with signed nearest/even rounding."""
    divisor = 1 << fractional_bits
    quotient, remainder = divmod(abs(product), divisor)
    if 2 * remainder > divisor or (2 * remainder == divisor and quotient % 2):
        quotient += 1
    return -quotient if product < 0 else quotient
