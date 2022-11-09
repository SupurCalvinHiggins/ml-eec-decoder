from collections import deque


class FeedForwardEncoder:
    def __init__(self, generator):
        self._generator = generator
        self._registers = deque([0] * (max(map(len, generator)) - 1))
        self._padding = [0] * (max(map(len, generator)) - 1)

    def encode(self, data):
        padded_data = data + self._padding
        output_data = []
        for bit in padded_data:
            for polynomial in self._generator:
                output_bit = bit
                for poly_bit, reg_bit in zip(polynomial[1:], self._registers):
                    if poly_bit:
                        output_bit ^= reg_bit
                output_data.append(output_bit)
            if self._registers:
                self._registers.rotate(1)
                self._registers[0] = bit
        return output_data
