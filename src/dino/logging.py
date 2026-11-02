"""Validated signal logging, independent of simulation scheduling."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .types import Array


@dataclass(frozen=True)
class SignalSpec:
    """A named signal's fixed vector width and unit label."""

    width: int
    unit: str

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be positive")


@dataclass(frozen=True)
class SimulationResult:
    """Copied read-only time (N,) and signals (N,width), including empty runs."""

    time: Array
    signals: Mapping[str, Array]
    specs: Mapping[str, SignalSpec]

    def __post_init__(self) -> None:
        t = np.array(self.time, dtype=float, copy=True)
        if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("time must be finite, rank-one, strictly increasing")
        if self.signals.keys() != self.specs.keys():
            raise ValueError("signals and specs must have identical keys")
        values = {}
        for name, signal in self.signals.items():
            a = np.array(signal, dtype=float, copy=True)
            if a.shape != (len(t), self.specs[name].width) or not np.all(np.isfinite(a)):
                raise ValueError(f"invalid shape or nonfinite signal: {name}")
            a.setflags(write=False)
            values[name] = a
        t.setflags(write=False)
        object.__setattr__(self, "time", t)
        object.__setattr__(self, "signals", MappingProxyType(values))
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))

    def to_csv(self, path: str | Path) -> None:
        """Export time and components with units in headers; overwrite path."""
        import csv

        with Path(path).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["time [s]"]
                + [
                    f"{name}[{i}] [{self.specs[name].unit}]"
                    for name in self.signals
                    for i in range(self.specs[name].width)
                ]
            )
            for k, t in enumerate(self.time):
                writer.writerow([t] + [v for a in self.signals.values() for v in a[k]])


class SignalLogger:
    """Append atomic samples; reject schema changes, duplicate times and NaN/Inf."""

    def __init__(self, specs: Mapping[str, SignalSpec]) -> None:
        self.specs = MappingProxyType(dict(specs))
        self._time: list[float] = []
        self._signals: dict[str, list[Array]] = {name: [] for name in specs}

    def append(self, time: float, signals: Mapping[str, Array]) -> None:
        """Copy one sample; all validation finishes before any storage changes."""
        if not np.isfinite(time) or (self._time and time <= self._time[-1]):
            raise ValueError("time must be finite and strictly increasing")
        if signals.keys() != self.specs.keys():
            raise ValueError("sample keys must match schema")
        sample = {k: np.array(v, dtype=float, copy=True) for k, v in signals.items()}
        for k, a in sample.items():
            if a.shape != (self.specs[k].width,) or not np.all(np.isfinite(a)):
                raise ValueError(f"invalid signal: {k}")
        self._time.append(time)
        for k, a in sample.items():
            self._signals[k].append(a)

    def result(self) -> SimulationResult:
        """Return an independent snapshot, including correctly shaped empty data."""
        return SimulationResult(
            np.array(self._time),
            {
                k: np.array(v).reshape(len(self._time), self.specs[k].width)
                for k, v in self._signals.items()
            },
            self.specs,
        )
