import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_height = 28
image_width = 28
batch_size = 2

model = keras.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10)
])

datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format='channels_first',
    validation_split=0.0,
    dtype=tf.float32,
)

train_generator = datagen.flow_from_directory(
    'Datasets/two_digit_train_images/',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='training',
    seed=123,
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_generator, epochs=2)