import numpy as np


class BinarySymmetricChannel:
    def __init__(self, crossover_probability: float) -> None:
        assert 0 <= crossover_probability <= 1
        self._crossover_probability = crossover_probability

    def transform(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.choice(
            [0, 1],
            size=data.size,
            p=[1 - self._crossover_probability, self._crossover_probability],
        )
        return np.logical_xor(data, noise)
