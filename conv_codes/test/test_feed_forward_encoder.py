import numpy as np
from conv_codes import FeedForwardEncoder


def test_feedforward_encoder_1() -> None:
    encoder = FeedForwardEncoder(transfer_matrix=[[1]])
    input_data = [1, 1, 0, 0, 1, 0]
    output_data = encoder.transform(input_data)
    assert np.array_equal(output_data, input_data)


def test_feedforward_encoder_10_11() -> None:
    encoder = FeedForwardEncoder(transfer_matrix=[[1, 0], [1, 1]])
    input_data = [1, 1, 0, 0, 1, 0]
    output_data = encoder.transform(input_data)
    assert np.array_equal(output_data, [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0])
