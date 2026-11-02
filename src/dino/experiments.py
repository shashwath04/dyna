"""Serial experiment orchestration; no dynamics or integration is performed here."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class Trial(Generic[T]):
    """Trial index, input parameters, and caller-defined output."""

    index: int
    parameters: Mapping[str, float]
    output: T


def parameter_sweep(
    grid: Mapping[str, Sequence[float]], run: Callable[[Mapping[str, float]], T]
) -> list[Trial[T]]:
    """Cartesian product in insertion order. Empty grid runs once; empty axis never.

    Caller exceptions propagate immediately; inputs to each run are independent.
    """
    names = list(grid)
    trials = []
    for i, values in enumerate(product(*(grid[k] for k in names))):
        parameters = dict(zip(names, values, strict=True))
        trials.append(Trial(i, dict(parameters), run(dict(parameters))))
    return trials


def monte_carlo(
    count: int,
    seed: int,
    sample: Callable[[np.random.Generator], Mapping[str, float]],
    run: Callable[[Mapping[str, float]], T],
) -> list[Trial[T]]:
    """Independent seeded streams per trial; repeatable for the same NumPy version.

    Sampling and model execution belong to the caller. Failures propagate.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")
    trials = []
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(count)):
        parameters = dict(sample(np.random.default_rng(child)))
        trials.append(Trial(i, dict(parameters), run(dict(parameters))))
    return trials
