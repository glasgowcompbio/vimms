import numpy as np

from vimms.ChineseRestaurantProcess import discrete_draw, Restricted_Crp
from vimms.Noise import (
    trunc_normal,
    GaussianPeakNoise,
    GaussianPeakNoiseLevelSpecific,
    UniformSpikeNoise,
)


def test_discrete_draw_deterministic():
    np.random.seed(0)
    result = discrete_draw([0.2, 0.8])
    assert result == 1


def test_restricted_crp_basic():
    np.random.seed(1)
    nxt, counts = Restricted_Crp(1, [1], [], 0)
    assert nxt == 0
    assert counts == [1]


def test_restricted_crp_existing_counts():
    np.random.seed(0)
    nxt, counts = Restricted_Crp(1, [1], [0], 1)
    assert nxt == 1
    assert counts == [1, 1]


def test_trunc_normal_positive():
    np.random.seed(0)
    val = trunc_normal(5, 1, False)
    assert val > 0
    np.random.seed(0)
    log_val = trunc_normal(5, 1, True)
    assert log_val > 0


def test_gaussian_peak_noise():
    np.random.seed(0)
    noise = GaussianPeakNoise(1)
    val = noise.get(10, 1)
    assert np.isclose(val, 11.764052345967665)


def test_gaussian_peak_noise_level_specific():
    np.random.seed(0)
    noise = GaussianPeakNoiseLevelSpecific({1: 1})
    val_level1 = noise.get(10, 1)
    val_level2 = noise.get(10, 2)
    assert np.isclose(val_level1, 11.764052345967665)
    assert val_level2 == 10


def test_uniform_spike_noise_sampling():
    np.random.seed(0)
    noise = UniformSpikeNoise(density=0.5, max_val=10, min_val=5)
    mzs, intensities = noise.sample(0, 10)
    assert len(mzs) == len(intensities) == 5
    assert np.all(mzs >= 0) and np.all(mzs <= 10)
    assert np.all(intensities >= 5) and np.all(intensities <= 10)


def test_uniform_spike_noise_bounds_override():
    np.random.seed(0)
    noise = UniformSpikeNoise(density=1, max_val=1, min_val=0, min_mz=5, max_mz=6)
    mzs, _ = noise.sample(0, 10)
    assert np.all(mzs >= 5) and np.all(mzs <= 6)
