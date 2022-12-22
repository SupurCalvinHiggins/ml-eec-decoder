import numpy as np


class BurstyBinarySymmetricChannel:
    def __init__(self, burst_probability: float, burst_length: int) -> None:
        """Construct a bursty binary symmetric channel. This channel introduces burst
        errors of burst_length bits with a probability of burst_probability. Note that
        this is NOT a channel model from the literature. However, similar channel models
        exist. See bursty binary noise channels.

        Args:
            burst_probability:
                The probabilty of a burst error occuring at any particular location.
            burst_length:
                The length of burst errors to generate. Each burst error will originate
                from a particular location and continue for burst_length bits.
        """

        assert 0 < burst_probability < 1
        assert burst_length > 0

        self._distribution = [1 - burst_probability, burst_probability]
        self._burst_length = burst_length

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Introduce burst errors into the data.

        Args:
            data:
                The array of bits to introduce burst errors to. The behavior of this
                method is undefined on arrays that contain elements other than 0 and 1.

        Returns: The data with burst errors.
        """

        # Convert the data to the appropriate data type.
        data = data.astype(np.uint8, copy=False)

        # Iterate starting from indices outside of the data. This accounts for burst
        # errors originating outside of the current message that could still impact
        # the current message.
        for i in range(-self._burst_length + 1, data.size):

            # Check if we should introduce a burst error.
            if not np.random.choice((0, 1), p=self._distribution):
                continue

            # Introduce burst error.
            l = max(i, 0)
            r = min(data.size - 1, i + self._burst_length)
            data[l:r] ^= 1

        return data
