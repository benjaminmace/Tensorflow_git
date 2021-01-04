import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pprint import pprint

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

pprint(model.variables)

class MyLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(MyLayer, self).__init__()
        self.my_var = tf.Variable(100)
        self.my_other_var_list = [tf.Variable(x) for x in range(2)]

m = MyLayer()

print([variable.numpy() for variable in m.variables])