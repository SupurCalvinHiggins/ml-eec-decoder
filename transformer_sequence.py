import numpy as np
import tensorflow as tf


class TransformerSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        encoder: object,
        channel: object,
        batch_shape: tuple[int, int, int, int],
    ) -> None:
        """Constructs a Sequence dataset yielding batches of data in the shape of
        batch_shape. Uniform random labels are generated in the shape of
        (symbol_count - encoder.pad_count,). These labels are transformed by the
        transformer to produce the dataset features. The features
        should are produced by transforming the random labels with the encoder and then
        the channel.

        Args:
            encoder:
                A convolutional code encoder.
            channel:
                A channel from the channels package.
            batch_shape:
                The shape of the batch in the format of
                (batch_count, batch_size, symbol_count, symbol_size) where batch_count
                is the number of batches generated per epoch, batch_size is the number
                of datum per batch, symbol_count is the number of symbols produced by
                the transformer, and symbol_size is the size of each symbol produced
                by the transformer.
        """
        self._encoder = encoder
        self._channel = channel
        (
            self._batch_count,
            self._batch_size,
            self._symbol_count,
            self._symbol_size,
        ) = batch_shape
        self._pad_count = encoder.pad_count
        self._X = None
        self._y = None
        self.on_epoch_end()

    def __len__(self) -> int:
        """Length of the sequence.

        Returns:
            The number of batches in the dataset.
        """
        return self._batch_count

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Gets a batch from the sequence.

        Returns:
            (X, y) of the batch at idx.
        """
        X_batch = self._X[idx, :, :, :]
        y_batch = self._y[idx, :, :, :]
        return X_batch.astype(np.uint8), y_batch.astype(np.uint8)

    def on_epoch_end(self) -> None:
        """Generates a new dataset"""

        # Generate random labels.
        y_shape = (self._batch_count, self._batch_size, self._symbol_count, 1)
        self._y = np.random.choice((0, 1), size=y_shape)

        # Set the pad symbols to 0.
        self._y[:, :, self._symbol_count - self._pad_count :] = 0

        # Allocate space for the features.
        X_shape = (
            self._batch_count,
            self._batch_size,
            self._symbol_count,
            self._symbol_size,
        )
        self._X = np.zeros(X_shape)

        # Generate the features.
        for i in range(self._batch_count):
            for j in range(self._batch_size):
                y_sample = self._y[i, j, : self._symbol_count - self._pad_count]
                X_sample_shape = (self._symbol_count, self._symbol_size)
                X_sample_encoded = self._encoder.transform(y_sample)
                X_sample_transmitted = self._channel.transform(X_sample_encoded)
                X_sample_reshaped = X_sample_transmitted.reshape(X_sample_shape)
                self._X[i, j, :, :] = X_sample_reshaped
