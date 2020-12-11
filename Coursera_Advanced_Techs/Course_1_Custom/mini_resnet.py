import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, \
    MaxPool2D, GlobalAveragePooling2D, Dense

import tensorflow_datasets as tfds

class IdentityBlock(Model):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name='')

        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)
        return x

class ResNet(Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = Conv2D(64, 7, padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')
        self.max_pool = MaxPool2D((3, 3))
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)
        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)
        return self.classifier(x)

def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

resnet = ResNet(10)

resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

dataset = tfds.load('mnist', split=tfds.Split.TRAIN)
dataset = dataset.map(preprocess).batch(32)


resnet.fit(dataset, epochs=1)