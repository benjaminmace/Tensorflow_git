import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class Model():
    def __init__(self):
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b

model = Model()

TRUE_w = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

random_xs = tf.random.normal(shape=[NUM_EXAMPLES])
ys = (TRUE_w * random_xs) + TRUE_b


def loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(y_true=outputs, y_pred=model(inputs))
    dw, db = tape.gradient(current_loss, [model.w, model.b])

    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    return current_loss


epochs = range(25)

for epoch in epochs:
    history = train(model=model, inputs=random_xs, outputs=ys, learning_rate=0.1)
    print(f'Epoch: {epoch}; Loss: {history}\n')
