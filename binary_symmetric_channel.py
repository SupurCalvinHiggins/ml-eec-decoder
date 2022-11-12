import numpy as np


class BinarySymmetricChannel:
    def __init__(self, crossover_probability: float) -> None:
        assert 0 <= crossover_probability <= 1
        self._distribution = [1 - crossover_probability, crossover_probability]
        self._sample_space = np.array([0, 1], dtype=np.bool8)

    def transform(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.bool8, copy=False)
        noise = np.random.choice(
            self._sample_space, size=data.size, p=self._distribution
        )
        return np.logical_xor(data, noise, dtype=np.bool8)
