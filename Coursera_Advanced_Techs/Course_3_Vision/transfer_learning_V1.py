import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)

(training_images, training_labels), \
(validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()

def preprocessing(input_images):
    input_images = input_images.astype('float16')
    output_images = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_images

training_images_processed = preprocessing(training_images)
validation_images_processed = preprocessing(validation_images)


def feature_extraction(inputs):
    feature_extractor_layer = tf.keras.applications.resnet.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet')(inputs)
    return feature_extractor_layer


def classifier(inputs):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(10, activation='softmax', name='classification')(x)
    return x

def final_model(inputs):
    resize = layers.UpSampling2D(size=(7, 7))(inputs)
    resnet = feature_extraction(resize)
    classification = classifier(resnet)

    return classification

inputs = layers.Input(shape=(32, 32, 3))
classification_output = final_model(inputs)

model = tf.keras.Model(inputs=inputs, outputs=classification_output)

model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_images_processed, training_labels,
                    epochs=3,
                    validation_data=(validation_images_processed, validation_labels),
                    batch_size=32)