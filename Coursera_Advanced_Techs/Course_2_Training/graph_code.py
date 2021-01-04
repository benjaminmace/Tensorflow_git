import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


@tf.function
def add(a, b):
    return a + b


#print(tf.autograph.to_code(add.python_function))


def linear_layer(x):
    return 2*x + 1

@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))

#print(deep_net(tf.constant((1, 2, 3))))


a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

result = f(1.0, 2.0)
#print(result)

@tf.function
def while_loop(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

print(tf.autograph.to_code(while_loop.python_function))