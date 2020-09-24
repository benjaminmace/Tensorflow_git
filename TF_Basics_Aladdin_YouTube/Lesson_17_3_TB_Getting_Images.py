import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import plot_to_image, image_grid

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)

(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True)

def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

def augment(image, label):
   if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

   image = tf.image.random_brightness(image, max_delta=0.1)
   image = tf.image.random_flip_left_right(image)

   image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

   return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE= 32

ds_train = ds_train.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog',
               'Frog', 'Horse', 'Ship', 'Truck']

def get_model():
    model = keras.Sequential(
        [
            layers.Input((32, 32, 3)),
            layers.Conv2D(8, 3, padding='same', activation='relu'),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(10),
        ]
    )
    return model

NUM_EPOCHS = 5
step = 0

model = get_model()

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
writer = tf.summary.create_file_writer('Lesson_17_3_Logs/train/')

for epoch in range(NUM_EPOCHS):
    for batch_idx, (x, y) in enumerate(ds_train):
        figure = image_grid(x, y, class_names)

        with writer.as_default():
            tf.summary.image(
                'Visualize Images', plot_to_image(figure), step=step
            )

            step += 1
