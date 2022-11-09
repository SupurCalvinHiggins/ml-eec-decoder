import numpy as np
from binary_symmetric_channel import BinarySymmetricChannel


def test_binary_symmetric_channel_0() -> None:
    np.random.seed(0)
    data = np.array([1, 0, 0, 1, 1])
    channel = BinarySymmetricChannel(crossover_probability=0.00)
    assert np.array_equal(channel.transform(data), data)


def test_binary_symmetric_channel_33() -> None:
    np.random.seed(0)
    data = np.random.choice([0, 1], size=100_000)
    channel = BinarySymmetricChannel(crossover_probability=0.33)
    assert abs(np.logical_xor(channel.transform(data), data).sum() - 33_333) < 100
