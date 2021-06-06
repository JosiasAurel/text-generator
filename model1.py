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

chars_val = dict((char, index) for index, char in enumerate(chars))

token_values = [i for i in chars_val.values()]

print(token_values)

model = keras.Sequential([
    keras.layers.LSTM(1)
])

model.compile(optimizer="sgd",
              loss=keras.losses.CategoricalCrossentropy(from_logits=False))


model.fit(token_values)
