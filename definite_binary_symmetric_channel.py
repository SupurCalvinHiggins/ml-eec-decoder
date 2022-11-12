import numpy as np


class DefiniteBinarySymmetricChannel:
    def __init__(self, flip_count: int) -> None:
        assert flip_count >= 0
        self._flip_count = flip_count

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert data.size >= self._flip_count
        data = data.astype(np.bool8, copy=False)
        idxs = np.random.choice(
            np.arange(data.size), size=self._flip_count, replace=False
        )
        data[idxs] = ~data[idxs]
        return data
