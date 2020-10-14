import tensorflow as tf
import os
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

BATCH_SIZE = 64
EPOCHS = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
PATIENCE = 10

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           zoom_range=0.5)

test_image_generator = ImageDataGenerator(rescale=1./255)

train_data = train_image_generator.flow_from_directory(directory=train_dir,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary',

                                                       )

test_data = train_image_generator.flow_from_directory(directory=validation_dir,
                                                       batch_size=BATCH_SIZE,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary')

def plot_images(arriving_image):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(arriving_image, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#sample_images, _ = next(train_data)

#plot_images(sample_images[:5])

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

history = model.fit(train_data,
                    batch_size=125,
                    epochs = EPOCHS,
                    validation_data = test_data,
                    callbacks=ES)


