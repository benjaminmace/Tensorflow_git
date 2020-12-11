import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense
from tensorflow.keras.models import Model


class CNNResidual(Layer):
    def __init__(self, layers, filters, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Conv2D(filters, (3,3), activation='relu') for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x


class DNNResidual(Layer):
    def __init__(self, layers, neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [Dense(neurons, activation='relu') for _ in range(layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return inputs + x


class MyResidual(Model):
    def __init__(self, **kwargs):
        self.hidden1 = Dense(30, activation='relu')
        self.block1 = CNNResidual(2, 32)
        self.block2 = DNNResidual(2, 64)
        self.out = Dense(1)

    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.block1(x)
        for _ in range(3):
            x = self.block2(x)
        return self.out(x)
