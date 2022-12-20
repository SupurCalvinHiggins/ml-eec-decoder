import numpy as np


class BurstChannel:
    def __init__(self, burst_probability: float, burst_length: int) -> None:
        assert 0 < burst_probability < 1
        assert burst_length > 0
        self._distribution = [1 - burst_probability, burst_probability]
        self._burst_length = burst_length

    def transform(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.uint8, copy=False)
        for i in range(-self._burst_length + 1, data.size):
            if not np.random.choice((0, 1), p=self._distribution):
                continue
            l = max(i, 0)
            r = min(data.size - 1, i + self._burst_length)
            data[l:r] ^= 1

        return data


if __name__ == "__main__":
    channel = BurstChannel(0.1, 5)
    for _ in range(20):
        data = np.ones(10)
        print(channel.transform(data))
