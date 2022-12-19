import numpy as np


class BurstChannel:
    def __init__(self, burst_probability: float, burst_length: int) -> None:
        assert 0 < burst_probability < 1
        assert burst_length > 0
        self._distribution = [1 - burst_probability, burst_probability]
        self._burst_length = burst_length

    def transform(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.bool8, copy=False)
        possible_burst_idxs = np.arange(data.size + (self._burst_length - 1))
        burst_idxs = np.random.choice(
            possible_burst_idxs, replace=False, p=self._distribution
        )
        for i in range(burst_idxs.size):
            idx = burst_idxs[i]
            l = max(idx, 0)
            r = idx + self._burst_length
            data[l:r] ^= 1
        return data
