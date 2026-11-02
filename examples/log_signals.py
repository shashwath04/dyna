"""Infrastructure-only example: record supplied data; no plant is simulated."""

from pathlib import Path

import numpy as np

from dino import SignalLogger, SignalSpec, plot_signal


def main() -> None:
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    logger = SignalLogger({"position": SignalSpec(1, "m")})
    for t, position in [(0.0, 0.0), (0.1, 0.2), (0.2, 0.3)]:
        logger.append(t, {"position": np.array([position])})
    result = logger.result()
    result.to_csv(output / "signals.csv")
    ax = plot_signal(result, "position")
    ax.figure.savefig(output / "signals.png")


if __name__ == "__main__":
    main()
