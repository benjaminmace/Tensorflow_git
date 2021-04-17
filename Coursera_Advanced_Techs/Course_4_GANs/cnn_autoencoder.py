import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import tensorflow_datasets as tfds
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.experimental.set_policy(policy)


def map_image(image, label):
  '''Normalizes and flattens the image. Returns image as input and label.'''
  image = tf.cast(image, dtype=tf.float16)
  image = image / 255.0

  return image, image

SHUFFLE_BUFFER_SIZE = 1024
BATCH_SIZE = 128

train_dataset = tfds.load('mnist', as_supervised=True, split="train")
train_dataset = train_dataset.map(map_image)
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()

def encoder(inputs):
    conv_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    max_pool_1 = layers.MaxPooling2D((2, 2))(conv_1)
    conv_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(max_pool_1)
    max_pool_2 = layers.MaxPooling2D((2, 2))(conv_2)

    return max_pool_2

def bottleneck(inputs):
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(inputs)
    encoder_vis = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(bottleneck)

    return bottleneck, encoder_vis

def decoder(inputs):
    conv_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    up_sample_1 = layers.UpSampling2D((2, 2))(conv_1)
    conv_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up_sample_1)
    up_sample_2 = layers.UpSampling2D((2, 2))(conv_2)
    conv_3 = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up_sample_2)

    return conv_3

def con_auto():
    inputs = layers.Input(shape=(28, 28, 1, ))

    encoder_output = encoder(inputs)

    bottleneck_output, encoder_vis = bottleneck(encoder_output)

    decoder_output = decoder(bottleneck_output)

    model = keras.Model(inputs=inputs, outputs=decoder_output)
    encoder_model = keras.Model(inputs=inputs, outputs=encoder_vis)

    return model, encoder_model

model , _ = con_auto()

model.compile(optimizer='adam', loss='binary_crossentropy')

train_steps = 60000 // BATCH_SIZE

model.fit(train_dataset, steps_per_epoch=train_steps, epochs=15)