"""Integer arithmetic reference from rational numbers; transaction scoreboard."""

from fractions import Fraction

import pytest

from dino import FixedPIDConfig, FPGAState, fixed_point_pid_step, fpga_controller_step

pytestmark = pytest.mark.student


@pytest.mark.parametrize("error", range(-15, 16))
@pytest.mark.parametrize("gain", [-7, -2, 1, 3, 6])
def test_product_rounding_against_rational_reference(error, gain):
    result = fixed_point_pid_step(error, 0, 0, gain, gain, 0, 2, 16, -32768, 32767)
    expected = round(Fraction(error * gain, 4))
    assert result == (expected, expected, error)


def test_wide_intermediates_saturate_only_at_registers():
    # Difference 127-(-128)=255 must not wrap to -1 in signed 8-bit arithmetic.
    assert fixed_point_pid_step(127, 120, -128, 127, 127, 127, 0, 8, -100, 100) == (100, 127, 127)
    assert fixed_point_pid_step(-128, -120, 127, 127, 127, 127, 0, 8, -100, 100) == (
        -100,
        -128,
        -128,
    )
    # Values above float's exact-integer range remain bit-accurate.
    raw = 2**58 + 3
    assert fixed_point_pid_step(raw, 0, 0, 1, 0, 0, 0, 64, -(2**63), 2**63 - 1)[0] == raw


@pytest.mark.parametrize(
    "f,w,error,lower,upper",
    [(7, 8, 0, -1, 1), (-1, 8, 0, -1, 1), (0, 8, 128, -1, 1), (0, 8, 0, 2, 1)],
)
def test_invalid_format_or_range(f, w, error, lower, upper):
    with pytest.raises(ValueError):
        fixed_point_pid_step(error, 0, 0, 1, 1, 0, f, w, lower, upper)


def test_clock_sequence_scoreboard():
    config = FixedPIDConfig(16, 4, 8, 4, 16, -20, 20)
    state = FPGAState()
    integral, previous, output = 0, 0, 0
    for error, valid, reset in [
        (8, True, False),
        (-4, True, False),
        (999999, False, False),
        (0, True, True),
        (12, True, False),
        (0, False, False),
    ]:
        state = fpga_controller_step(error, state, config, valid, reset)
        if reset:
            output, integral, previous = 0, 0, 0
        elif valid:
            output = min(
                20,
                max(
                    -20,
                    round(Fraction(error * 16, 16))
                    + integral
                    + round(Fraction((error - previous) * 8, 16)),
                ),
            )
            integral += round(Fraction(error * 4, 16))
            previous = error
        assert state == FPGAState(output, integral, previous, valid and not reset)
