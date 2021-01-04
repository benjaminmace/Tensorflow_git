import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

x = tf.ones((2, 2))
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.square(y)

dz_dx = t.gradient(z, x)



a = tf.constant(3.0)
with tf.GradientTape(persistent=True) as r:
    r.watch(a)
    b = a * a
    c = b * b
dc_da = r.gradient(c, a)
db_da = r.gradient(b, a)


del r

