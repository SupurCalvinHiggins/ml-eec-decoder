import numpy as np


class DefiniteBinarySymmetricChannel:
    def __init__(self, flip_count: int) -> None:
        """Construct a definite binary symmetric channel. This channel introduces a
        fixed number of bit errors into a message at uniform random locations. Note
        that this is NOT a channel model from the literature. This channel is primarily
        used for model testing and debugging.

        Args:
            flip_count: The number of bits to flip in the message.
        """

        assert flip_count >= 0

        self._flip_count = flip_count

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Introduce a definite number of bit errors into the provided data.

        Args:
            data: The data to introduce errors to. The behavior of this
            method is undefined on arrays that contain elements other than 0 and 1.

        Return: The data after errors have been introduced.
        """

        assert data.size >= self._flip_count

        # Convert the data to the appropriate type.
        data = data.astype(np.uint8, copy=False)

        # Choose the indices of the bits to flip.
        idxs = np.random.choice(
            np.arange(data.size),
            size=self._flip_count,
            replace=False,
        )

        # Flip the bits.
        data[idxs] = ~data[idxs] & 1

        return data
