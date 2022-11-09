import time
import numpy as np
from feed_forward_encoder import FeedForwardEncoder


BENCHMARK_SIZE = 10_000_000


def make_benchmark(generator):
    def benchmark() -> int:
        np.random.seed(0)
        data = list(np.random.choice([0, 1], size=BENCHMARK_SIZE))
        encoder = FeedForwardEncoder(generator=generator)
        start_time = time.time()
        _ = encoder.encode(data)
        end_time = time.time()
        return end_time - start_time

    return benchmark


def benchmark_all() -> None:
    generators = [
        [[1, 0], [1, 1]],
        [[1, 0, 0], [1, 0, 1], [1, 1, 1]],
        [
            [1, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
        ],
    ]
    for generator in generators:
        print(f"Benchmark {generator}: {make_benchmark(generator)()} seconds")


if __name__ == "__main__":
    benchmark_all()
