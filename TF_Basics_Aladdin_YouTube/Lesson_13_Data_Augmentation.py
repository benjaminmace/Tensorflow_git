import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_datasets as tfds

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

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE= 32

def augment(image, label):
    new_height = new_width = 32
    image = tf.image.resize(image, (new_height, new_width))

    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    image = tf.image.random_flip_left_right(image)

    return image, label

ds_train = ds_train.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
#ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)


data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(height=32, width=32),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
        layers.experimental.preprocessing.RandomContrast(factor=0.1)
    ])

model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    #data_augmentation,
    layers.Conv2D(64, 3,
                  padding='same',
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3,
                  padding='same',
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Conv2D(256, 3,
                  padding='same',
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Conv2D(512, 3,
                  padding='same',
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),
    layers.Conv2D(1024, 3,
                  padding='same',
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)),
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(2048,
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1024,
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(512,
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(ds_train, epochs=7)
model.evaluate(ds_test)
