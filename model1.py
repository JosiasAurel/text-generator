import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import tensorflow as tf

with open("./tweets/lfact.txt") as f:
    _data = f.read().splitlines()
    data = list(set(_data))

_train = []
train = []

for i in data:
    sp = i.split(" ")
    _train.append(sp)

for i in _train:
    for j in i:
        train.append(str(j))

chars = sorted(set(train))

""" chars_val = dict((char, index) for index, char in enumerate(chars))

token_values = [i for i in chars_val.values()]

print(token_values)

model = keras.Sequential([
    keras.layers.LSTM(1)
])

model.compile(optimizer="sgd",
              loss=keras.losses.CategoricalCrossentropy(from_logits=False))


model.fit(token_values)
 """

# the unique texts int eh vocabulary of the model
vocabulary = sorted(set(chars))

# split the text into tokens
chars = tf.strings.unicode_split(vocabulary, input_encoding="UTF-8")

# STringLookup object
char_ids_from = StringLookup(vocabulary=list(vocabulary), mask_token=None)

# convert each character of the vocabulary to have unique text IDs
ids = char_ids_from(chars)

ids_chars_from = StringLookup(
    vocabulary=char_ids_from.get_vocabulary(), invert=True, mask_token=None)

chars = ids_chars_from(ids)

""" Do not worry about the [UNK] for now, they are just some non UTF-8 chars like ß """
# print(chars)

print(chars)


def text_from_ids(ids):
    return tf.strings.reduce_join(ids_chars_from(ids), axis=-1)


all_char_ids = char_ids_from(tf.strings.unicode_split(data, "UTF-8"))

ids_dataset = tf.data.Dataset.from_tensor_slices(all_char_ids)


# The fuck am i doing since 9am 😭
