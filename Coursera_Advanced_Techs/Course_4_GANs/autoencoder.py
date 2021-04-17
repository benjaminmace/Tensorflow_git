import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import numpy as np
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.experimental.set_policy(policy)


def generate_data(m):
    '''plots m random points on a 3D plane'''

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles) / 2 + 0.1 * np.random.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + 0.1 * np.random.randn(m) / 2
    data[:, 2] = data[:, 0] * 0.1 + data[:, 1] * 0.3 + 0.1 * np.random.randn(m)

    return data

data = generate_data(100)
data = data - data.mean(axis=0, keepdims=0)

ax = plt.axes(projection='3d')
ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], cmap='Reds')

encoder = tf.keras.models.Sequential([layers.Dense(2, input_shape=[3])])
decoder = tf.keras.models.Sequential([layers.Dense(3, input_shape=[2])])

autoencoder = tf.keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1.5))

history = autoencoder.fit(data, data, epochs=200)

codings = encoder.predict(data)
print(f'input point: {data[0]}')
print(f'encoded point: {codings[0]}')

fig = plt.figure(figsize=(8,6))
plt.plot(codings[:, 0], codings[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()

decodings = decoder.predict(codings)

print(f'input point: {data[0]}')
print(f'encoded point: {codings[0]}')
print(f'decoded point: {decodings[0]}')

ax = plt.axes(projection='3d')
ax.scatter3D(decodings[:, 0], decodings[:, 1], decodings[:, 2], c=decodings[:, 0], cmap='Reds')
