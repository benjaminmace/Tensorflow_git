import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = keras.Sequential()

model.add(keras.Input(shape=(None, 28)))
model.add(layers.SimpleRNN(256, return_sequences=True, activation='relu'))
model.add(layers.SimpleRNN(256, activation='relu'))
model.add(layers.Dense(10))

#model_2 = keras.Sequential()

#model_2.add(keras.Input(shape=(None, 28)))
#model_2.add(layers.GRU(512, return_sequences=True, activation='tanh'))
#model_2.add(layers.GRU(512, return_sequences=True, activation='tanh'))
#model_2.add(layers.Dense(10))

model_3 = keras.Sequential()

model_3.add(keras.Input(shape=(None, 28)))
model_3.add(layers.LSTM(512,
                        dropout=0.2,
                        return_sequences=True,
                        activation='tanh'))
model_3.add(layers.LSTM(512,
                        dropout=0.2,
                        activation='tanh'))
model_3.add(layers.Dense(10))


ES = keras.callbacks.EarlyStopping(monitor='accuracy', patience=4)
model_3.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

model_3.fit(x_train, y_train, batch_size=64, epochs=75)
model_3.evaluate(x_test, y_test, batch_size=64)