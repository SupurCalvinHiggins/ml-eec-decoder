import numpy as np


class FeedForwardEncoder:
    def __init__(self, generator):
        self._generator = generator
        self._generator_degree = len(generator[0])
        self._padding = np.array([0] * (self._generator_degree - 1))
        assert all(map(lambda x: len(x) == self._generator_degree, self._generator))

    def encode(self, data):
        input_data = np.append(data, self._padding)
        output_data = np.zeros(input_data.size * len(self._generator))

        for poly_bits in zip(*self._generator):
            for i, poly_bit in enumerate(poly_bits):
                if poly_bit:
                    output_data[i :: len(self._generator)] = np.logical_xor(
                        output_data[i :: len(self._generator)],
                        input_data,
                    )
            input_data = np.roll(input_data, 1)
            input_data[0] = 0

        return output_data
