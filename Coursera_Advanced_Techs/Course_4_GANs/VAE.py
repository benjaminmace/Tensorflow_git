import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import numpy as np
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.experimental.set_policy(policy)


def map_image(image, label):
    image = tf.cast(image, dtype=tf.float16)
    image = image / 255.0
    image = tf.reshape(image, shape=(28, 28, 1,))

    return image

SHUFFLE_BUFFER_SIZE = 1024
BATCH_SIZE = 128

train_dataset = tfds.load('mnist', as_supervised=True, split="train")
train_dataset = train_dataset.map(map_image)
train_dataset = train_dataset.shuffle(1024).batch(128)

test_dataset = tfds.load('mnist', as_supervised=True, split="test")
test_dataset = test_dataset.map(map_image)
test_dataset = test_dataset.shuffle(1024).batch(128)

def encoder_layers(inputs, latent_dim):
    x = layers.Conv2D(32,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      activation='relu',
                      name='encode_conv1')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64,
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      activation='relu',
                      name='encode_conv2')(x)
    batch_2 = layers.BatchNormalization()(x)

    x = layers.Flatten(name='encode_flatten')(batch_2)
    x = layers.Dense(20, activation='relu', name='encode_dense')(x)
    x = layers.BatchNormalization()(x)

    mu = layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = layers.Dense(latent_dim, name='latent_sigma')(x)

    return mu, sigma, batch_2.shape


class Sampling(layers.Layer):
    def call(self, inputs, **kwargs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), dtype=tf.float16)
        return mu + tf.exp(0.5 * sigma) * epsilon


def encoder_model(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    mu, sigma, conv_shape = encoder_layers(inputs, latent_dim=latent_dim)
    z = Sampling()((mu, sigma))
    model = keras.Model(inputs, outputs=[mu, sigma, z])
    return model, conv_shape


def decoder_layers(inputs, conv_shape):
    units = conv_shape[1] * conv_shape[2] * conv_shape[3]

    x = layers.Dense(units,
                     activation='relu',
                     name='decode_dense1')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name='decode_reshape')(x)
    x = layers.Conv2DTranspose(filters=64,
                               kernel_size=2,
                               strides=2,
                               padding='same',
                               activation='relu',
                               name='decode_conv2d_2')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=32,
                               kernel_size=3,
                               strides=2,
                               padding='same',
                               activation='relu',
                               name='decode_conv2d_3')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(filters=1,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               activation='sigmoid',
                               name='decode_final')(x)

    return x


def decoder_model(latent_dim, conv_shape):
    inputs = layers.Input(shape=(latent_dim,))
    outputs = decoder_layers(inputs, conv_shape)
    model = keras.Model(inputs, outputs)
    return model


def kl_reconstruction_loss(inputs, outputs, mu, sigma):
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss) * -0.5
    return kl_loss

def vae_model(encoder, decoder, input_shape):
    inputs = layers.Input(shape=input_shape)
    mu, sigma, z = encoder(inputs)
    reconstructed = decoder(z)
    model = keras.Model(inputs=inputs, outputs=reconstructed)
    loss = kl_reconstruction_loss(inputs, z, mu, sigma)
    model.add_loss(loss)
    return model


def get_models(input_shape, latent_dim):
    encoder, conv_shape = encoder_model(input_shape=input_shape, latent_dim=latent_dim)
    decoder = decoder_model(latent_dim=latent_dim, conv_shape=conv_shape)
    vae = vae_model(encoder, decoder, input_shape=input_shape)

    return encoder, decoder, vae


LATENT_DIM = 2

encoder, decoder, vae = get_models(input_shape=(28,28,1,), latent_dim=LATENT_DIM)

random_vector_for_generation = tf.random.normal(shape=[16, LATENT_DIM])

optimizer = tf.keras.optimizers.Adam()
loss_metric = tf.keras.metrics.Mean()
bce_loss = tf.keras.losses.BinaryCrossentropy()

for epoch in range(100):
    print('Start of epoch %d' % (epoch,))

    # iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:

            # feed a batch to the VAE model
            reconstructed = vae(x_batch_train)

            # compute reconstruction loss
            flattened_inputs = tf.reshape(x_batch_train, shape=[-1])
            flattened_outputs = tf.reshape(reconstructed, shape=[-1])
            loss = bce_loss(flattened_inputs, flattened_outputs) * 784

            # add KLD regularization loss
            loss += sum(vae.losses)

            # get the gradients and update the weights
        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        # compute the loss metric
        loss_metric(loss)

        # display outputs every 100 steps
        if step % 100 == 0:
            print('Epoch: %s step: %s mean loss = %s' % (epoch, step, loss_metric.result().numpy()))