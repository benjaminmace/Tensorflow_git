import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1

    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    plt.figure(figsize=(n_cols, n_rows))

    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")


(X_train, _), _ = keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32) / 255.

BATCH_SIZE = 128

dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)

random_normal_dimensions = 32

generator = keras.models.Sequential(
    [
        layers.Dense(64, activation='selu', input_shape=[random_normal_dimensions]),
        layers.Dense(128, activation='selu'),
        layers.Dense(28 * 28, activation='sigmoid'),
        layers.Reshape([28, 28]),
])

discriminator = keras.models.Sequential(
    [
        layers.Flatten(input_shape=[28, 28]),
        layers.Dense(128, activation='selu'),
        layers.Dense(64, activation='selu'),
        layers.Dense(1, activation='sigmoid'),

    ]
)

discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
discriminator.trainable = False

gan = keras.models.Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='rmsprop')

def train_gan(gan, dataset, random_normal_dims, n_epochs=50):
    generator, discriminator = gan.layers

    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}/{n_epochs}')
        for real_images in dataset:
            #infer batch size
            batch_size = real_images.shape[0]

            #PHASE 1 - train discriminator
            noise = tf.random.normal(shape=[batch_size, random_normal_dims])
            fake_images = generator(noise)

            #combine fake and real images
            mixed_images = tf.concat([fake_images, real_images], axis=0)

            #creat the labels for the discriminator
            discriminator_labels = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(mixed_images, discriminator_labels)

            #PHASE 2 - try to fool discriminator
            noise = tf.random.normal(shape=[batch_size, random_normal_dims])
            generator_labels = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, generator_labels)

        plot_multiple_images(fake_images, 8)
        plt.show(block=False)
        plt.pause(interval=2)
        plt.close()

train_gan(gan, dataset, random_normal_dimensions, n_epochs=20)