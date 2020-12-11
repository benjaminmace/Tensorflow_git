import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class SimpleDense(Layer):

    def __init__(self, units=32, activation=None):
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name='kernel',
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                             trainable=True)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name='bias',
                             initial_value=b_init(shape=(self.units,), dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)


xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model = tf.keras.Sequential([SimpleDense(units=1)])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))