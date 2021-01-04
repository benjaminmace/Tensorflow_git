import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random

x_train = np.array([-1., 0.0, 1., 2., 3., 4.], dtype=float)
y_train = np.array([-3., -1., 1., 3., 5., 7.], dtype=float)

w = tf.Variable(random.random(), trainable=True)
b = tf.Variable(random.random(), trainable=True)

def simple_loss(real_y, pred_y):
    return tf.abs(real_y - pred_y)

LEARNING_RATE = 0.01

def fit_data(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        pred_y = w * real_x + b
        reg_loss = simple_loss(real_y, pred_y)

    w_gradient = tape.gradient(reg_loss, w)
    b_gradient = tape.gradient(reg_loss, b)

    w.assign_sub(w_gradient * LEARNING_RATE)
    b.assign_sub(b_gradient + LEARNING_RATE)

for _ in range(500):
    fit_data(x_train, y_train)

print(f'y ≈ {w.numpy()}x + {b.numpy()} ')