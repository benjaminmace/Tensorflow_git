import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

NUM_EPOCHS = 1

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()


for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    train_step = test_step = 0
    train_writer = tf.summary.create_file_writer('Lesson_17_Logs/train/'+str(lr))
    test_writer = tf.summary.create_file_writer('Lesson_17_Logs/test/'+str(lr))
    model = get_model()
    optimizer = keras.optimizers.Adam(lr=lr)

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (x, y) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn(y, y_pred)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc_metric.update_state(y, y_pred)

            with train_writer.as_default():
                tf.summary.scalar('Loss', loss, step=train_step)
                tf.summary.scalar('Accuracy', acc_metric.result(), step=train_step)
                train_step += 1

        acc_metric.reset_states()

        for batch_idx, (x, y) in enumerate(ds_test):
            y_pred = model(x, training=False)
            loss = loss_fn(y, y_pred)
            acc_metric.update_state(y, y_pred)

            with test_writer.as_default():
                tf.summary.scalar('Loss', loss, step=test_step)
                tf.summary.scalar('Accuracy', acc_metric.result(), step=test_step)

                test_step += 1

        acc_metric.reset_states()