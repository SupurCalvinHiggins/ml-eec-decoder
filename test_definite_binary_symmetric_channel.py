import numpy as np
from definite_binary_symmetric_channel import DefiniteBinarySymmetricChannel


def test_definite_binary_symmetric_channel_0() -> None:
    np.random.seed(0)
    data = np.array([1, 0, 0, 1, 1])
    channel = DefiniteBinarySymmetricChannel(flip_count=0)
    assert np.logical_xor(channel.transform(data), data).sum() == 0


def test_definite_binary_symmetric_channel_3() -> None:
    np.random.seed(0)
    data = np.array([1, 0, 0, 1, 1])
    channel = DefiniteBinarySymmetricChannel(flip_count=3)
    assert np.logical_xor(channel.transform(data), data).sum() == 3
