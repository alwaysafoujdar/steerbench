import numpy as np
import pytest

from steerbench.config import MachineConfig


@pytest.fixture
def cfg() -> MachineConfig:
    return MachineConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
