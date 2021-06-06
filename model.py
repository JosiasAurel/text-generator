# from tensorflow import keras
import numpy as np

raw_texts = open("./tweets/lfact.txt").read().lower()

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
    seq_out = raw_texts[i + seq_len]
    data_x.append([chars_to_int[char] for char in seq_in])
    data_y.append(chars_to_int[seq_out])

n_patterns = len(data_x)

# reshape samples to have the shape [sample, time_steps, features]
X = np.reshape(data_x, (n_patterns, seq_len, 1))

# normalize the data
X = X / float(n_vocab)

# one-hot encode output variable
y = np_utils.to_categorical(data_y)
