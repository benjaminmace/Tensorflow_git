import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt



URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)

print(path_to_zip)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

print(PATH)

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=.20,
                                           height_shift_range=.20,
                                           shear_range=0.2,
                                           zoom_range=0.5,
                                           horizontal_flip=True)

test_image_generator = ImageDataGenerator(rescale=1./255)

train_data = train_image_generator.flow_from_directory(directory=train_dir,
                                                       batch_size=20,
                                                       shuffle=True,
                                                       target_size=(150, 150),
                                                       class_mode='binary')

test_data = test_image_generator.flow_from_directory(directory=validation_dir,
                                                     batch_size=20,
                                                     target_size=(150, 150),
                                                     class_mode='binary')


pre_trained_model = InceptionV3(include_top=False,
                                weights='imagenet',
                                input_shape=(150, 150, 3),
                                pooling=None,
                                classes=1000,
                                classifier_activation='softmax')

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')

last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    validation_data=test_data,
                    steps_per_epoch=100,
                    epochs=25,
                    validation_steps=50)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
