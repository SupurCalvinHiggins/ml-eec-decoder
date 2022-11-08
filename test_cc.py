from cc import CC, FeedForwardEncoder


def test_cc_encode() -> None:
    data = [1, 1, 0, 0, 1, 0, 1]
    result = CC().encode(data)
    assert result == [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1]


def test_feedforward_encoder_1() -> None:
    encoder = FeedForwardEncoder(generator=[[1]])
    input_data = [1, 1, 0, 0, 1, 0]
    output_data = encoder.encode(input_data)
    assert output_data == input_data


def test_feedforward_encoder_10_11() -> None:
    encoder = FeedForwardEncoder(generator=[[1, 0], [1, 1]])
    input_data = [1, 1, 0, 0, 1, 0]
    output_data = encoder.encode(input_data)
    assert output_data == [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
