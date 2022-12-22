from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Input, Bidirectional
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from conv_codes import FeedForwardEncoder
from channels import BurstyBinarySymmetricChannel
from transformer_sequence import TransformerSequence
from viterbi_decoder import ViterbiDecoder
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
        batch_shape=(64, 32, DATA_SYMBOL_COUNT + encoder.pad_count, SYMBOL_SIZE),
        pad_count=encoder.pad_count,
    )
    return sequence


def main() -> None:
    for burst_length in range(1, 32):
        sequence = build_sequence(burst_length)
        model_path = os.path.join(
            "lstm_burst_model_output", f"burst_length_{burst_length}.h5"
        )
        model = load_model(model_path)

        baseline_norms = []
        model_norms = []
        for i in range(len(sequence)):
            X_batch, y_batch = sequence[i]
            for j in range(X_batch.shape[0]):
                print(
                    f"{burst_length}: {i*X_batch.shape[0] + j}/{len(sequence) * X_batch.shape[0]}"
                )
                decoder = ViterbiDecoder(TRANSFER_MATRIX)
                X_sample = X_batch[j].flatten()
                y_sample = y_batch[j].flatten()
                y_baseline = decoder.decode(X_sample)
                y_model = model.predict(X_batch[j][np.newaxis, :])
                y_baseline[-2:] = 0
                baseline_norm = (
                    (y_sample.astype(np.uint8) ^ y_baseline.astype(np.uint8)) ** 2
                ).sum()
                baseline_norms.append(baseline_norm)
                model_norm = (
                    (
                        y_sample.astype(np.uint8)
                        ^ np.around(y_model[0].flatten()).astype(np.uint8)
                    )
                    ** 2
                ).sum()
                model_norms.append(model_norm)

        csv_path = os.path.join("lstm_burst_model_output", "out_with_std.csv")
        with open(csv_path, "a") as f:
            f.writelines(
                [
                    f"{burst_length},{np.mean(baseline_norms)},{np.std(baseline_norms)},{np.mean(model_norms)},{np.std(model_norms)}\n"
                ]
            )

        csv_path2 = os.path.join("lstm_burst_model_output", "baseline_norms.csv")
        with open(csv_path2, "a") as f:
            f.writelines(
                [f"{burst_length}," + ",".join(str(x) for x in baseline_norms)]
            )

        csv_path3 = os.path.join("lstm_burst_model_output", "model_norms.csv")
        with open(csv_path3, "a") as f:
            f.writelines([f"{burst_length}," + ",".join(str(x) for x in model_norms)])


if __name__ == "__main__":
    main()
