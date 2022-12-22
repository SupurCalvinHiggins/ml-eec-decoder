import numpy as np


class BinarySymmetricChannel:
    def __init__(self, crossover_probability: float) -> None:
        """Construct a binary symmetric channel. This channel introduces single bit
        errors with crossover_probability probability.

        Args:
            crossover_probability:
                The probabilty of a single bit error.
        """

        assert 0 <= crossover_probability <= 1

        self._distribution = [1 - crossover_probability, crossover_probability]

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Introduce bit errors into the data.

        Args:
            data:
                The array of bits to introduce bit errors to. The behavior of this
                method is undefined on arrays that contain elements other than 0 and 1.

        Return: The data with bit errors.
        """

        # Convert the data to the appropriate data type.
        data = data.astype(np.uint8, copy=False)

        # Generate noise.
        bits = np.array([0, 1], dtype=np.uint8)
        noise = np.random.choice(bits, size=data.size, p=self._distribution)

        # Introduce noise into the data.
        return data ^ noise
