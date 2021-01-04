import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

strategy = tf.distribute.MirroredStrategy()

def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predications = model(images, training=True)
        loss = compute_loss(labels, predications)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predications)
    return loss

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                           per_replica_losses, axis=None)

EPOCHS = 10
for epoch in EPOCHS:
    total_loss = 0.
    num_batches = 0
    for batch in train_dist_dataset:
        total_loss += distributed_train_step(batch)
        num_batches += 1
    train_loss = total_loss / num_batches

