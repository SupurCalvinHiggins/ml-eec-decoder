import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    LSTM,
    Dense,
    TimeDistributed,
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
)
from feed_forward_encoder import FeedForwardEncoder


# model.add(Conv1D(filters=10, kernel_size=2, activation="relu"))

model = Sequential()
model.add(Input(shape=(64, 2)))
model.add(Bidirectional(LSTM(units=8, return_sequences=True)))
model.add(TimeDistributed(Dense(units=1, activation="sigmoid")))
model.compile(loss="mse", optimizer="adam")
model.summary()

# pred = model.predict(np.random.choice([0, 1], size=128).reshape(1, 64, 2))
# print(pred.shape)
# pred = model.predict(np.random.choice([0, 1], size=256).reshape(1, 256, 1))
# print(pred.shape)
# exit()

encoder = FeedForwardEncoder([[1, 0, 0], [1, 0, 1]])
Y = np.random.choice([0, 1], size=(1000, 64, 1))
Y[:, 62:, :] = 0
X = np.array(
    [encoder.encode(Y[i, :62, :].reshape(62)).reshape(64, 2) for i in range(1000)]
)

model.fit(X, Y, epochs=20)


y_test = np.random.choice([0, 1], size=64)
y_test[62:] = 0
x_test = encoder.encode(y_test[:62].reshape(62))
pred_prob = model.predict(x_test.reshape(1, 64, 2))[0].reshape(64)
pred = np.around(pred_prob)
print(pred_prob)
print(pred)
print(y_test)
print(np.logical_xor(y_test, pred))


# model.fit(x=x.reshape(1, 128, 1), y=y.reshape(1, 62, 1))

# https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/
# https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

# https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM
# https://goodboychan.github.io/python/deep_learning/tensorflow-keras/2020/12/09/01-RNN-Many-to-many.html
# https://stackabuse.com/solving-sequence-problems-with-lstm-in-keras-part-2/
# https://stackabuse.com/solving-sequence-problems-with-lstm-in-keras/
# https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM
#
#


class CCSequence(tf.keras.utils.Sequence):
    def __init__(self, cc, batches, batch_size):
        self.cc = cc
        self.batches = batches
        self.batch_size = batch_size
        self.generate_xy()

    def generate_xy(self):
        size = self.batches * self.batch_size * 62
        self.y = np.random.choice([0, 1], size=size)
        self.x = np.array(
            [self.cc.encode(self.y[idx : idx + 62]) for idx in range(0, size, 62)]
        )
        # print("self.x.shape", self.x.shape)
        # print("self.y.shape", self.y.shape)

    def __len__(self):
        return self.batches * self.batch_size

    def __getitem__(self, idx):
        # print("idx", idx)
        batch_x = self.x[idx].reshape(1, 128)
        batch_y = self.y[idx * 62 : (idx + 1) * 62].reshape(1, 62)
        # print("batch_x.shape", batch_x.shape)
        # print("batch_y.shape", batch_y.shape)
        # OBSERVATIONS MIGHT BE IN THE ROWS
        assert np.array_equal(self.cc.encode(batch_y), batch_x.flatten())
        batch_y_padded = np.append(batch_y, [0, 0])
        batch_x_shaped = batch_x.reshape(1, 64, 2)
        # print("batch_x_shaped.shape", batch_x_shaped.shape)
        # print("batch_y_padded.shape", batch_y_padded.shape)
        return batch_x_shaped, batch_y_padded
