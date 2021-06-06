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
