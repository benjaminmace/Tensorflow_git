import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np

train_data = tfds.load('fashion_mnist', split='train')
test_data = tfds.load('fashion_mnist', split='test')


def format_image(data):
    return (tf.cast(tf.reshape(data['image'], [-1]), 'float32')) / 255., data['label']


train_data = train_data.map(format_image)
test_data = test_data.map(format_image)

train = train_data.batch(batch_size=128)
test = test_data.batch(batch_size=128)

def base_model():
    inputs = tf.keras.Input(shape=(784,), name='clothing')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = tf.keras.optimizers.Adam()

NUM_EPOCHS = 25

model = base_model()


def apply_gradient(optimizer, model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss_object(y, logits)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return logits, loss_value


def train_data_for_one_epoch():
    losses = []
    for step, (x_batch_train, y_batch_train) in enumerate(train):
        logits, loss_value = apply_gradient(optimizer, model, x_batch_train, y_batch_train)
        losses.append(loss_value)
        train_acc_metric.update_state(y_batch_train, logits)
    return losses


def perform_validation():
    losses = []
    for x_val, y_val in test:
        val_logits = model(x_val)
        val_loss = loss_object(y_val, val_logits)
        losses.append(val_loss)
        val_acc_metric.update_state(y_val, val_logits)
    return losses


for epoch in range(NUM_EPOCHS+1):
    losses_train = train_data_for_one_epoch()
    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()

    losses_val = perform_validation()
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

    losses_train_mean = np.mean(losses_train)
    losses_val_mean = np.mean(losses_val)

    print(f'Epoch: {epoch}; Training Loss: {losses_train_mean:.4f}; Training Accuracy: {train_acc:.4f}; '
          f'Validation Loss: {losses_val_mean:.4f}; Validation Accuracy: {val_acc:.4f}\n')

