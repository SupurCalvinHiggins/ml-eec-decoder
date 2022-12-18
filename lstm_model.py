from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Input, Bidirectional
from feed_forward_encoder import FeedForwardEncoder
from binary_symmetric_channel import BinarySymmetricChannel
from transformer_pipeline import TransformerPipeline
from transformer_sequence import TransformerSequence


TRANSFER_MATRIX = [[1, 1, 0], [1, 0, 1]]
CROSSOVER_PROBABILITY = 0.1
DATA_SYMBOL_COUNT = 62
SYMBOL_SIZE = len(TRANSFER_MATRIX)


def build_sequence() -> TransformerSequence:
    encoder = FeedForwardEncoder(transfer_matrix=TRANSFER_MATRIX)
    channel = BinarySymmetricChannel(crossover_probability=CROSSOVER_PROBABILITY)
    pipeline = TransformerPipeline(transformers=[encoder, channel])
    sequence = TransformerSequence(
        transformer=pipeline,
        batch_shape=(32, 32, DATA_SYMBOL_COUNT + encoder.pad_count, SYMBOL_SIZE),
        pad_count=encoder.pad_count,
    )
    return sequence


def build_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=(None, SYMBOL_SIZE)))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(TimeDistributed(Dense(units=1, activation="sigmoid")))
    model.compile(loss="mse", optimizer="adam")
    return model


def main() -> None:
    sequence = build_sequence()
    model = build_model()
    model.summary()
    model.fit(sequence, epochs=1000, verbose=1)
    results = model.evaluate(sequence, verbose=1)
    print(results)


# 4 stacked lstm units=64 layers
# crossover 0.1, batchsize/count = 32,32
# converged after 70 epoch to loss 0.17 mse
# w/ transfer_matrix=[[1, 1, 0], [1, 0, 1]]
# high error floor because complete symbol flip is likely


if __name__ == "__main__":
    main()
