import numpy as np


class FeedForwardEncoder:
    def __init__(self, transfer_matrix: list[list[int]]) -> None:
        """Constructs a feedforward (FIR) convolutional encoder from a given transfer
        function matrix with polynomial entries in F[[x]]^2. See Chapter 12 of
        Todd Moon's Error Correction Coding: Mathematical Methods and Algorithms for
        additional details.

        Args:
            transfer_matrix:
                A transfer function matrix with polynomial entries in F[[x]]^2. The
                polynomials are represented by a list of coefficients. For example, the
                transfer function matrix [1 | 1 + x + x^3] is represented by the list
                [[1, 0, 0, 0], [1, 1, 0, 1]]. The number of coefficients in each list
                must be the same, i.e., leading zero coefficients must be provided. At
                least one of the polynomial representations must have maximum degree,
                i.e., at least one of the lists must end in a 1.
        """
        self._transfer_matrix = transfer_matrix
        self._transfer_matrix_degree = len(transfer_matrix[0]) - 1
        self._padding = np.array([0] * self._transfer_matrix_degree)
        assert all(
            len(poly) == len(self._transfer_matrix[0]) for poly in self._transfer_matrix
        )
        assert any(poly[-1] != 0 for poly in self._transfer_matrix)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes an array of bits using the feedforward convolutional encoder.
        Additional zero bits equivalent to the maximum degree of the entries of
        transfer function matrix will be appended to the data prior to encoding to
        ensure proper termination of the encoder.

        Args:
            data: Array of bits to encode with the feedforward convolutional encoder.

        Return: The array of bits after encoding.
        """
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
