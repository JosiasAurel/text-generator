from tensorflow import keras
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import callbacks

raw_texts = open("wonder.txt").read().lower()

unique_chars = sorted(list(set(raw_texts)))

chars_to_int = dict((value, index) for index, value in enumerate(unique_chars))

n_chars = len(raw_texts)
n_vocab = len(unique_chars)

# preparing the model for input and output pairs
# the data will be in sequences of 100
seq_len = 100
data_x = []
data_y = []

for i in range(0, n_chars - seq_len, 1):
    seq_in = raw_texts[i:i + seq_len]
    seq_out = raw_texts[i+seq_len]
    # print([char for char in seq_in])
    data_x.append([chars_to_int[char] for char in seq_in])
    data_y.append(chars_to_int[seq_out])

n_patterns = len(data_x)

# reshape samples to have the shape [sample, time_steps, features]
X = np.reshape(data_x, (n_patterns, seq_len, 1))

# normalize the data
X = X / float(n_vocab)

# one-hot encode output variable
y = np_utils.to_categorical(data_y)

# the model -> LSTM
model = keras.Sequential([
    keras.layers.LSTM(256, input_shape=(X.shape[1], X.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=y.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy")

"""
# define model checkpoint to not lose training line
filepath = "model-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor="loss", verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
"""

int_to_char = dict((index, value) for index, value in enumerate(unique_chars))
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
