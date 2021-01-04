import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

a = tf.constant([[5, 7], [2, 1]])
b = tf.add(a,2)
c = b ** 2
d = tf.reduce_sum(c)
print(d)

e = 49 + 81 + 16 + 9
print(e)


import sys
sys.exit()

def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))