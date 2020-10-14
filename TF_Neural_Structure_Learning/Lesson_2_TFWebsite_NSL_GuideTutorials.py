import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import neural_structured_learning as nsl

print('Version: ', tf.__version__)
print('Eager mode: ', tf.executing_eagerly())
print("GPU is",
      'available' if tf.config.list_physical_devices("GPU") else 'Not available!')