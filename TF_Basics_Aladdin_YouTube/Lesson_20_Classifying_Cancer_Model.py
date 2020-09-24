import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_examples = 20225
test_examples = 2555
validation_examples = 2551
img_height = img_width = 224
batch_size = 32

#model = keras.Sequential([
#    hub.KerasLayer('https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4', trainable=True),
#    layers.Dense(1, activation='sigmoid')
#])

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last',
    dtype=tf.float32,
)

validation_datagen = ImageDataGenerator(rescale=1.0/255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0/255, dtype=tf.float32)

train_gen = train_datagen.flow_from_directory(
    'ISIC/data/train/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=123
)

validation_gen = validation_datagen.flow_from_directory(
    'ISIC/data/validation/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=123
)

test_gen = test_datagen.flow_from_directory(
    'ISIC/data/test/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=123
)

METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]

model = tf.keras.models.load_model('Lesson_20_Saved_Model/')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=METRICS
)

model.fit(
    train_gen,
    epochs=2,
    steps_per_epoch=250,
    validation_data=validation_gen,
    validation_steps=75,
    callbacks=[keras.callbacks.ModelCheckpoint('Lesson_20_Saved_Model/isic_model')]
)

#model.evaluate(validation_gen)
#model.evaluate(test_gen)
