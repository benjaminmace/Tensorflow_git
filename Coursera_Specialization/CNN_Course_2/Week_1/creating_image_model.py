import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EPOCHS = 2
IMG_HEIGHT = 150
IMG_WIDTH = 150
PATIENCE = 10

DIR = 'E:\\Code\\Projects\\Images\\Cats_and_Dogs\\cats_and_dogs_filtered'

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           zoom_range=0.5)

test_image_generator = ImageDataGenerator(rescale=1./255)

train_data = train_image_generator.flow_from_directory(directory=DIR+'\\train',
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='binary',

                                                       )

test_data = train_image_generator.flow_from_directory(directory=DIR+'\\validation',
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
model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
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


