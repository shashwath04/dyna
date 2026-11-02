import csv

import numpy as np
import pytest

import dino


def test_public_api():
    for name in dino.__all__:
        assert getattr(dino, name) is not None


def test_logger_snapshots_and_atomicity(tmp_path):
    logger = dino.SignalLogger({"x": dino.SignalSpec(2, "m")})
    assert logger.result().signals["x"].shape == (0, 2)
    value = np.array([1.0, 2.0])
    logger.append(0.0, {"x": value})
    value[:] = 9
    first = logger.result()
    for time, signals in [(0.0, {"x": value}), (1.0, {}), (1.0, {"x": np.array([np.nan, 0.0])})]:
        with pytest.raises(ValueError):
            logger.append(time, signals)
    logger.append(1.0, {"x": value})
    assert first.time.shape == (1,)
    np.testing.assert_array_equal(first.signals["x"], [[1.0, 2.0]])
    with pytest.raises(ValueError):
        first.signals["x"][0, 0] = 5
    path = tmp_path / "data.csv"
    first.to_csv(path)
    with path.open() as stream:
        assert list(csv.reader(stream)) == [
            ["time [s]", "x[0] [m]", "x[1] [m]"],
            ["0.0", "1.0", "2.0"],
        ]


def test_result_validation():
    with pytest.raises(ValueError):
        dino.SimulationResult(np.array([1.0, 0.0]), {}, {})
    with pytest.raises(ValueError):
        dino.SignalSpec(0, "m")


def test_sweep_and_rng():
    result = dino.parameter_sweep({"a": [1.0, 2.0], "b": [3.0, 4.0]}, lambda p: p["a"] + p["b"])
    assert [t.output for t in result] == [4.0, 5.0, 5.0, 6.0]
    assert len(dino.parameter_sweep({}, lambda p: p)) == 1
    assert dino.parameter_sweep({"a": []}, lambda p: p) == []

    def sample(rng):
        return {"x": float(rng.normal())}

    a = dino.monte_carlo(4, 42, sample, lambda p: p["x"])
    assert a == dino.monte_carlo(4, 42, sample, lambda p: p["x"])
    assert len({t.output for t in a}) == 4
    assert dino.monte_carlo(0, 42, sample, lambda p: p) == []
    with pytest.raises(ValueError):
        dino.monte_carlo(-1, 42, sample, lambda p: p)
    with pytest.raises(ZeroDivisionError):
        dino.parameter_sweep({}, lambda p: 1 / 0)


def test_plot(tmp_path):
    import matplotlib.pyplot as plt

    logger = dino.SignalLogger({"x": dino.SignalSpec(1, "m")})
    logger.append(0.0, {"x": np.array([2.0])})
    ax = dino.plot_signal(logger.result(), "x")
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "x [m]"
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2.0])
    ax.figure.savefig(tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
    plt.close(ax.figure)
