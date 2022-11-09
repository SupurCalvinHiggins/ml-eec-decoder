import numpy as np


class FeedForwardEncoder:
    def __init__(self, transfer_matrix: list[list[int]]) -> None:
        """Construct a feedforward (FIR) convolutional encoder from a given transfer
        function matrix with polynomial entries in F[[x]]^2.

        Args:
            transfer_matrix:
                A transfer function matrix with polynomial entries in F[[x]]^2. The
                polynomials are represented by a list of coefficients. For example, the
                transfer function matrix [1 | 1 + x + x^3] is represented by the list
                [[1, 0, 0, 0], [1, 1, 0, 1]]. The number of coefficients in each list
                must be the same, i.e., leading zero coefficients must be provided.
        """
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
