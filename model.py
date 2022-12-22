import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import (
    LSTM,
    Dense,
    TimeDistributed,
    Input,
    Conv1D,
    MaxPooling1D,
    Bidirectional,
)
from conv_codes import FeedForwardEncoder
from channels.binary_symmetric_channel import BinarySymmetricChannel


# model.add(Conv1D(filters=10, kernel_size=2, activation="relu"))

# prob_single_valid = 0.95**2
# prob_single_invalid = 1 - prob_single_valid
# expected_invalid = 0
# for num_invalid in range(0, 65):
#     prob = prob_single_invalid ** (num_invalid)
#     expected_invalid += prob * num_invalid
# print(expected_invalid)
# exit()


model = Sequential()
model.add(Input(shape=(None, 2)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(TimeDistributed(Dense(units=1, activation="sigmoid")))
model.compile(loss="mse", optimizer="adam")
model.summary()

encoder = FeedForwardEncoder([[1, 1, 0], [1, 0, 1]])
# model.fit(
#     EncoderSequence(
#         encoder=encoder,
#         batches=1024,
#         x_shape=(64, 2),
#         y_size=62,
#         padding_size=2,
#         batch_size=32,
#         channel=BinarySymmetricChannel(crossover_probability=0.01),
#     ),
#     epochs=10,
# )
# model.save("model.h5")
model = load_model("model.h5")

test_encoder = EncoderSequence(
    encoder=encoder,
    batches=1,
    x_shape=(128, 2),
    y_size=126,
    padding_size=2,
    batch_size=1,
    channel=BinarySymmetricChannel(crossover_probability=0),
)

for i in range(64):
    x_test, y_test = test_encoder[0]
    x_test = np.array(x_test, dtype=np.uint8)
    x_test[:, i, :] ^= 1
    pred_prob = model.predict(x_test)[0].reshape(128)
    pred = np.array(np.around(pred_prob), dtype=np.uint8)
    y_test = y_test.reshape(128)
    # print(pred_prob)
    # print(pred)
    # print(y_test)
    print(np.bitwise_xor(y_test, pred).sum() / y_test.size)


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


# class CCSequence(tf.keras.utils.Sequence):
#     def __init__(self, cc, batches, batch_size):
#         self.cc = cc
#         self.batches = batches
#         self.batch_size = batch_size
#         self.generate_xy()

#     def generate_xy(self):
#         size = self.batches * self.batch_size * 62
#         self.y = np.random.choice([0, 1], size=size)
#         self.x = np.array(
#             [self.cc.encode(self.y[idx : idx + 62]) for idx in range(0, size, 62)]
#         )
#         # print("self.x.shape", self.x.shape)
#         # print("self.y.shape", self.y.shape)

#     def __len__(self):
#         return self.batches * self.batch_size

#     def __getitem__(self, idx):
#         # print("idx", idx)
#         batch_x = self.x[idx].reshape(1, 128)
#         batch_y = self.y[idx * 62 : (idx + 1) * 62].reshape(1, 62)
#         # print("batch_x.shape", batch_x.shape)
#         # print("batch_y.shape", batch_y.shape)
#         # OBSERVATIONS MIGHT BE IN THE ROWS
#         assert np.array_equal(self.cc.encode(batch_y), batch_x.flatten())
#         batch_y_padded = np.append(batch_y, [0, 0])
#         batch_x_shaped = batch_x.reshape(1, 64, 2)
#         # print("batch_x_shaped.shape", batch_x_shaped.shape)
#         # print("batch_y_padded.shape", batch_y_padded.shape)
#         return batch_x_shaped, batch_y_padded
