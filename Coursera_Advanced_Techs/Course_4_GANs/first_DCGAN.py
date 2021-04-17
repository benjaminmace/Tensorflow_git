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


(X_train, _), _ = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32) / 255
X_train = X_train.reshape(-1, 28, 28, 1) * 2. - 1.

BATCH_SIZE = 128
codings_size = 32

dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)

def plot_results(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1

    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    plt.figure(figsize=(n_cols, n_rows))

    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")


generator = keras.models.Sequential(
    [
        layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
        layers.Reshape([7, 7, 128]),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='selu'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding='same', activation='tanh')
    ]
)

discriminator = keras.models.Sequential(
    [
        layers.Conv2D(64,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      activation=keras.layers.LeakyReLU(0.2),
                      input_shape=[28, 28, 1]),
        layers.Dropout(0.4),
        layers.Conv2D(128,
                      kernel_size=5,
                      strides=2,
                      padding='same',
                      activation=keras.layers.LeakyReLU(0.2)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ]
)

gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False

gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

def train_gan(gan, dataset, random_norm_dim, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        for real_images in dataset:
            batch_size = real_images.shape[0]
            noise = tf.random.normal(shape=[batch_size, random_norm_dim])
            fake_images = generator(noise)
            mixed_images = tf.concat([fake_images, real_images], axis=0)
            discriminator_labels = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(mixed_images, discriminator_labels)
            noise = tf.random.normal(shape=[batch_size, random_norm_dim])
            generator_labels = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, generator_labels)
        plot_results(fake_images, 16)
        plt.show(block=False)
        plt.pause(interval=1)
        plt.close()

train_gan(gan, dataset, codings_size, 100)