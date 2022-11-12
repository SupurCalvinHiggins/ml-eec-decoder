import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, Input
from feed_forward_encoder import FeedForwardEncoder


model = Sequential()
model.add(LSTM(units=4, input_shape=(128, 1), return_sequences=True))
# model.add(RepeatVector(64))
model.add(LSTM(units=4, input_shape=(64, 1), return_sequences=True))
model.add(TimeDistributed(Dense(2, activation="softmax")))
model.compile(loss="binary_crossentropy", optimizer="adam")
model.summary()

pred = model.predict(np.random.choice([0, 1], size=128).reshape(1, 128, 1))
print(pred.shape)
# lstm = LSTM(units=4, input_shape=(128, 1), return_sequences=False)
# repeat = RepeatVector(4)(np.array([0, 1, 2]).reshape(1, 3))
# print(repeat)
exit()

# model = Sequential()
# model.add(Input(shape=128))
# model.add(Dense(128))
# model.add(Dense(128))
# model.add(Dense(62, activation="sigmoid"))
# model.compile(loss="mean_squared_error")
# pred = model.predict(np.random.choice([0, 1], size=128).reshape(1, 128, 1))
# print(pred.shape)


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
        print("self.x.shape", self.x.shape)
        print("self.y.shape", self.y.shape)

    def __len__(self):
        return self.batches * self.batch_size

    def __getitem__(self, idx):
        print("idx", idx)
        batch_x = self.x[idx].reshape(1, 128)
        batch_y = self.y[idx * 62 : (idx + 1) * 62].reshape(1, 62)
        print("batch_x.shape", batch_x.shape)
        print("batch_y.shape", batch_y.shape)
        # OBSERVATIONS MIGHT BE IN THE ROWS
        assert np.array_equal(self.cc.encode(batch_y), batch_x.flatten())
        batch_y = np.array([[0, 1] if x else [1, 0] for x in batch_y[0]])
        return batch_x, batch_y


encoder = FeedForwardEncoder([[1, 0, 0], [1, 0, 1]])
model.fit(x=CCSequence(encoder, batches=10, batch_size=10))


y = np.random.choice([0, 1], size=62)
x = encoder.encode(y)
pred = model.predict(x.reshape(1, 128, 1))
print(pred[0])
print(y)
# model.fit(x=x.reshape(1, 128, 1), y=y.reshape(1, 62, 1))

# https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/
# https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

# https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM
#
#
#
#
#
#
