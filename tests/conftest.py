import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture
def scalar_estimate():
    from dino import Estimate

    return Estimate(np.array([0.0]), np.array([[1.0]]))
