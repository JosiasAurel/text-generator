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
