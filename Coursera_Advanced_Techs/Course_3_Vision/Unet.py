import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset, info = tfds.load('oxford_iiit_pet', with_info=True)

OUTPUT_CHANNELS = 3
EPOCHS = 25
BUFFER = 3000
BATCH_SIZE = 32

def random_flip(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128), method='nearest')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method='nearest')
    input_image, input_mask = random_flip(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128), method='nearest')
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method='nearest')
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def conv2d_block(input_tensor, n_filters, kernel_size=3):
    x = input_tensor

    for i in range(2):
        x = layers.Conv2D(filters=n_filters,
                          kernel_size=(kernel_size, kernel_size),
                          kernel_initializer='he_normal',
                          padding='same')(x)
        x = layers.Activation('relu')(x)

    return x


def encoder_block(inputs, n_filters, pool_size, dropout):
    f = conv2d_block(inputs, n_filters=n_filters)
    p = layers.MaxPooling2D(pool_size=pool_size)(f)
    p = layers.Dropout(dropout)(p)

    return f, p


def encoder(inputs):
    f1, p1 = encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3)
    f2, p2 = encoder_block(p1, n_filters=128, pool_size=(2, 2), dropout=0.3)
    f3, p3 = encoder_block(p2, n_filters=256, pool_size=(2, 2), dropout=0.3)
    f4, p4 = encoder_block(p3, n_filters=512, pool_size=(2, 2), dropout=0.3)

    return p4, (f1, f2, f3, f4)


def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=1024)

    return bottle_neck


def decoder_block(inputs, conv_output, n_filters, kernel_size, strides, dropout):
    u = layers.Conv2DTranspose(n_filters,
                               kernel_size,
                               strides=strides,
                               padding='same')(inputs)
    c = layers.concatenate([u, conv_output])
    c = layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)

    return c


def decoder(inputs, convs, output_channels):
    f1, f2, f3, f4 = convs

    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)

    outputs = layers.Conv2D(output_channels, (1, 1), activation='softmax')(c9)

    return outputs


def unet():
    inputs = layers.Input(shape=(128, 128, 3,))
    encoder_output, convs = encoder(inputs)

    bottle_neck = bottleneck(encoder_output)

    outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

model = unet()

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

model.load_weights('Unet.h5')

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

TRAIN_LENGTH = info.splits['train'].num_examples
VAL_SUBSPLITS = 5
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model.fit(train_dataset,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS,
          validation_data=test_dataset)

model.save('Unet.h5')