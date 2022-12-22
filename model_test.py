import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, Input
from conv_codes import FeedForwardEncoder


X = [x for x in range(5, 301, 5)]
Y = [y for y in range(20, 316, 5)]

X = np.array(X).reshape(20, 3, 1)
Y = np.array(Y).reshape(20, 3, 1)

model = Sequential()
model.add(LSTM(100, activation="relu", input_shape=(3, 1)))
model.add(RepeatVector(3))
model.add(LSTM(100, activation="relu", return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer="adam", loss="mse")
print(model.summary())

history = model.fit(X, Y, epochs=1000, validation_split=0.2, verbose=1, batch_size=100)
test_input = np.array([300, 305, 310])
test_input = test_input.reshape((1, 3, 1))
test_output = model.predict(test_input, verbose=0)
print(test_output)
