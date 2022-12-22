import time
import numpy as np
from conv_codes import FeedForwardEncoder


BENCHMARK_SIZE = 10_000_000


def make_benchmark(transfer_matrix):
    def benchmark() -> int:
        np.random.seed(0)
        data = np.random.choice([0, 1], size=BENCHMARK_SIZE)
        encoder = FeedForwardEncoder(transfer_matrix=transfer_matrix)
        start_time = time.time()
        _ = encoder.encode(data)
        end_time = time.time()
        return end_time - start_time

    return benchmark


def benchmark_all() -> None:
    transfer_matrices = [
        [[1, 0], [1, 1]],
        [[1, 0, 0], [1, 0, 1], [1, 1, 1]],
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
        ],
    ]
    for transfer_matrix in transfer_matrices:
        print(
            f"Benchmark {transfer_matrix}: {make_benchmark(transfer_matrix)()} seconds"
        )


if __name__ == "__main__":
    benchmark_all()
