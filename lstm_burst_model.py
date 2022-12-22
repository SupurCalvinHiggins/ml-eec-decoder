from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Input, Bidirectional
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from conv_codes import FeedForwardEncoder
from channels import BurstyBinarySymmetricChannel
from transformer_sequence import TransformerSequence
from conv_codes import ViterbiDecoder
import numpy as np
import os
import csv


TRANSFER_MATRIX = [[1, 1, 1], [1, 0, 1]]
CROSSOVER_PROBABILITY = 0.04
DATA_SYMBOL_COUNT = 64 - (len(TRANSFER_MATRIX[0]) - 1)
SYMBOL_SIZE = len(TRANSFER_MATRIX)


def build_sequence(burst_length: int) -> TransformerSequence:
    encoder = FeedForwardEncoder(transfer_matrix=TRANSFER_MATRIX)
    channel = BurstyBinarySymmetricChannel(CROSSOVER_PROBABILITY, burst_length)
    sequence = TransformerSequence(
        encoder=encoder,
        channel=channel,
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
    model.compile(loss="mse", optimizer="nadam")
    return model


def approximate_baseline_loss(sequence: TransformerSequence) -> float:
    total_error = 0
    for i in range(len(sequence)):
        X_batch, y_batch = sequence[i]
        for j in range(X_batch.shape[0]):
            decoder = ViterbiDecoder(TRANSFER_MATRIX)
            X_sample = X_batch[j].flatten()
            y_sample = y_batch[j].flatten()
            y_baseline = decoder.decode(X_sample)
            y_baseline[-2:] = 0
            norm = (
                (y_sample.astype(np.uint8) ^ y_baseline.astype(np.uint8)) ** 2
            ).sum()
            total_error += norm

    sequence.on_epoch_end()

    samples = len(sequence) * sequence[0][0].size
    return total_error / samples


def main() -> None:
    for burst_length in range(1, 32):
        sequence = build_sequence(burst_length)

        baseline_loss = approximate_baseline_loss(sequence)
        print("*" * 100)
        print(f"Approximate baseline loss: {baseline_loss}")
        print("*" * 100)

        model = build_model()
        model.summary()

        early_stopping = EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5)
        model.fit(
            sequence,
            epochs=1000,
            callbacks=[reduce_lr, early_stopping],
            verbose=1,
        )
        model_path = os.path.join(
            "lstm_burst_model_output", f"burst_length_{burst_length}.h5"
        )
        model.save(model_path)
        results = model.evaluate(sequence, verbose=1)
        print("*" * 100)
        print(f"Evaluate loss: {results}")
        print("*" * 100)

        csv_path = os.path.join("lstm_burst_model_output", "out.csv")
        with open(csv_path, "a") as f:
            f.writelines([f"{burst_length},{baseline_loss},{results}\n"])


# 4 stacked lstm units=64 layers
# crossover 0.1, batchsize/count = 32,32
# converged after 70 epoch to loss 0.17 mse
# w/ transfer_matrix=[[1, 1, 0], [1, 0, 1]]
# high error floor because complete symbol flip is likely


if __name__ == "__main__":
    main()
