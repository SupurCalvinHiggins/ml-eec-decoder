import numpy as np


class FeedForwardEncoder:
    def __init__(self, transfer_matrix: list[list[int]]) -> None:

        self._transfer_matrix = transfer_matrix
        self._transfer_matrix_degree = len(transfer_matrix[0])
        self._padding = np.array([0] * (self._transfer_matrix_degree - 1))
        assert all(
            map(lambda x: len(x) == self._transfer_matrix_degree, self._transfer_matrix)
        )

    def encode(self, data):
        input_data = np.append(data, self._padding)
        output_data = np.zeros(input_data.size * len(self._transfer_matrix))

        for poly_bits in zip(*self._transfer_matrix):
            for i, poly_bit in enumerate(poly_bits):
                if poly_bit:
                    output_data[i :: len(self._transfer_matrix)] = np.logical_xor(
                        output_data[i :: len(self._transfer_matrix)],
                        input_data,
                    )
            input_data = np.roll(input_data, 1)
            input_data[0] = 0

        return output_data
