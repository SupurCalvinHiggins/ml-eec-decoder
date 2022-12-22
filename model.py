import os
import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Input, Bidirectional
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from conv_codes import FeedForwardEncoder, FeedForwardDecoder
from channels import BurstyBinarySymmetricChannel
from transformer_sequence import TransformerSequence


# Encoder and channel constants.
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


def baseline_loss_estimate(sequence: TransformerSequence) -> float:
    total_error = 0
    for i in range(len(sequence)):
        X_batch, y_batch = sequence[i]
        for j in range(X_batch.shape[0]):
            decoder = FeedForwardDecoder(TRANSFER_MATRIX)
            X_sample = X_batch[j].flatten()
            y_sample = y_batch[j].flatten()
            y_baseline = decoder.transform(X_sample)
            y_baseline[-2:] = 0
            norm = (
                (y_sample.astype(np.uint8) ^ y_baseline.astype(np.uint8)) ** 2
            ).sum()
            total_error += norm

    sequence.on_epoch_end()

    samples = len(sequence) * sequence[0][0].size
    return total_error / samples


def train_models(output_directory: str) -> None:
    for burst_length in range(1, 32):

        # Generate a new dataset for the burst length.
        sequence = build_sequence(burst_length)

        # Compute and display an estimate of the baseline loss.
        baseline_loss = baseline_loss_estimate(sequence)
        print("*" * 100)
        print(f"Baseline loss: {baseline_loss}")
        print("*" * 100)

        # Display the model.
        model = build_model()
        model.summary()

        # Set up callbacks.
        early_stopping = EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5)

        # Train the model.
        model.fit(
            sequence,
            epochs=1,
            callbacks=[reduce_lr, early_stopping],
            verbose=1,
        )

        # Save the model.
        model_name = f"model_l{burst_length}.h5"
        model_path = os.path.join(output_directory, model_name)
        model.save(model_path)

        # Compute and display an estimate of the model loss.
        results = model.evaluate(sequence, verbose=1)
        print("*" * 100)
        print(f"Evaluate loss: {results}")
        print("*" * 100)

        # Compute and save model and baseline mean and std Hamming distance.
        print("Testing model...")
        baseline_norms = []
        model_norms = []
        for _ in range(2):
            sequence.on_epoch_end()
            for i in range(len(sequence)):
                X_batch, y_batch = sequence[i]
                for j in range(X_batch.shape[0]):
                    decoder = FeedForwardDecoder(TRANSFER_MATRIX)
                    X_sample = X_batch[j].flatten()
                    y_sample = y_batch[j].flatten()
                    y_baseline = decoder.transform(X_sample)
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
                print(f"Batch {i}/{len(sequence)}")

        csv_path = os.path.join(output_directory, "out.csv")
        with open(csv_path, "a") as f:
            f.writelines(
                [
                    f"{burst_length},{np.mean(baseline_norms)},{np.std(baseline_norms)},{np.mean(model_norms)},{np.std(model_norms)}\n"
                ]
            )


def main() -> None:

    # Create the output directory.
    now = datetime.datetime.now()
    output_directory = os.path.join(
        "models",
        now.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(output_directory)

    # Train the models.
    train_models(output_directory)


if __name__ == "__main__":
    main()
