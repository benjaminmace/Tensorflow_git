import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import os
import numpy as np

train_dir = 'E:\Code\Projects\Education\Tensorflow_git\Coursera_Specialization\CNN_Course_2\Week_4\RPS\\train'
test_dir = 'E:\Code\Projects\Education\Tensorflow_git\Coursera_Specialization\CNN_Course_2\Week_4\RPS\\test'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   zoom_range=0.5)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(300, 300),
                                                    batch_size=128,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(300, 300),
                                                  batch_size=128,
                                                  class_mode='categorical'
                                                  )


model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['acc'])

model.fit(train_generator,
          epochs=20,
          validation_data=test_generator,
          workers=15,
          max_queue_size=100)

root_dir = 'RPS/val/'

for _, _, files in os.walk(root_dir):
    for file in files:
        img = tf.keras.preprocessing.image.load_img(os.path.join(root_dir, file),
                                              color_mode='rgb',
                                              target_size=(300, 300))

        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])

        classes = model.predict(images)

        print(file)
        if np.argmax(classes) == 0:
            print('Paper')
        elif np.argmax(classes) == 1:
            print('Rock')
        else: print('Scissors')

