"""End-to-end scalar LQR example; CSV/plot outputs for pipeline smoke checks."""

from pathlib import Path

import numpy as np

from dino import Array, lqr, plot_signal, simulate


def main() -> None:
    # Plant x_dot=u. Unit costs give analytic K=1 and ideal continuous x=e^-t.
    gain = lqr(np.zeros((1, 1)), np.ones((1, 1)), np.eye(1), np.eye(1))

    def controller(t: float, state: Array, reference: Array) -> Array:
        return -gain @ (state - reference)

    result = simulate(
        lambda t, x, u: u,
        controller,
        np.ones(1),
        np.zeros(1),
        duration=2.0,
        integration_dt=0.01,
        control_dt=0.1,
    )
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    result.to_csv(output / "closed_loop.csv")
    ax = plot_signal(result, "state")
    ax.figure.savefig(output / "closed_loop.png")
    print(f"Recorded {result.time.size} samples; final state={result.signals['state'][-1, 0]:.6f}")


if __name__ == "__main__":
    main()
