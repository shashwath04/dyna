"""Matplotlib helpers; importing Dino never opens a window."""

from typing import TYPE_CHECKING

from .logging import SimulationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_signal(result: SimulationResult, name: str, ax: "Axes | None" = None) -> "Axes":
    """Plot each component against seconds; caller owns saving, showing and closing."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for i in range(result.specs[name].width):
        ax.plot(result.time, result.signals[name][:, i], label=f"{name}[{i}]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{name} [{result.specs[name].unit}]")
    ax.legend()
    return ax
