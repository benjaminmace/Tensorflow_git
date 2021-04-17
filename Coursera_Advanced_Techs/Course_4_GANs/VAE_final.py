import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import numpy as np
import matplotlib.pyplot as plt
import random
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# set a random seed
np.random.seed(51)

# parameters for building the model and training
BATCH_SIZE=125
LATENT_DIM=512
IMAGE_SIZE=64

# Data Preparation Utilities

def get_dataset_slice_paths(image_dir):
    image_file_list = os.listdir(image_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    return image_paths


def map_image(image_filename):
    img_raw = tf.io.read_file(image_filename)
    image = tf.image.decode_jpeg(img_raw)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image = tf.reshape(image, shape=(IMAGE_SIZE, IMAGE_SIZE, 3,))

    return image

# get the list containing the image paths
paths = get_dataset_slice_paths("E:\Code\Projects\Education\Tensorflow_git\Coursera_Advanced_Techs\Course_4_GANs\\anime/images/")

# shuffle the paths
random.shuffle(paths)

# split the paths list into to training (80%) and validation sets(20%).
paths_len = len(paths)
train_paths_len = int(paths_len * 0.8)

train_paths = paths[:train_paths_len]
val_paths = paths[train_paths_len:]

# load the training image paths into tensors, create batches and shuffle
training_dataset = tf.data.Dataset.from_tensor_slices((train_paths))
training_dataset = training_dataset.map(map_image)
training_dataset = training_dataset.shuffle(1000).batch(BATCH_SIZE)

# load the validation image paths into tensors and create batches
validation_dataset = tf.data.Dataset.from_tensor_slices((val_paths))
validation_dataset = validation_dataset.map(map_image)
validation_dataset = validation_dataset.batch(BATCH_SIZE)


print(f'number of batches in the training set: {len(training_dataset)}')
print(f'number of batches in the validation set: {len(validation_dataset)}')

def display_faces(dataset, size=9):
    '''Takes a sample from a dataset batch and plots it in a grid.'''
    dataset = dataset.unbatch().take(size)
    n_cols = 3
    n_rows = size//n_cols + 1
    plt.figure(figsize=(5, 5))
    i = 0
    for image in dataset:
        i += 1
        disp_img = np.reshape(image, (64,64,3))
        plt.subplot(n_rows, n_cols, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(disp_img)


def display_one_row(disp_images, offset, shape=(28, 28)):
    '''Displays a row of images.'''
    for idx, image in enumerate(disp_images):
        plt.subplot(3, 10, offset + idx + 1)
        plt.xticks([])
        plt.yticks([])
        image = np.reshape(image, shape)
        plt.imshow(image)


def display_results(disp_input_images, disp_predicted):
    '''Displays input and predicted images.'''
    plt.figure(figsize=(15, 5))
    display_one_row(disp_input_images, 0, shape=(IMAGE_SIZE,IMAGE_SIZE,3))
    display_one_row(disp_predicted, 20, shape=(IMAGE_SIZE,IMAGE_SIZE,3))


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        """Generates a random sample and combines with the encoder output

        Args:
        inputs -- output tensor from the encoder

        Returns:
          `inputs` tensors combined with a random sample
        """

        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = mu + tf.exp(0.5 * sigma) * epsilon
        return z


def encoder_layers(inputs, latent_dim):
    """Defines the encoder's layers.
    Args:
    inputs -- batch from the dataset
    latent_dim -- dimensionality of the latent space

    Returns:
    mu -- learned mean
    sigma -- learned standard deviation
    batch_3.shape -- shape of the features before flattening
    """
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation='relu',
                               name="encode_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu',
                               name="encode_conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu',
                               name="encode_conv3")(x)
    batch_3 = tf.keras.layers.BatchNormalization()(x)

    # flatten the features and feed into the Dense network
    x = tf.keras.layers.Flatten(name="encode_flatten")(batch_3)

    # we arbitrarily used 20 units here but feel free to change and see what results you get
    x = tf.keras.layers.Dense(1024, activation='relu', name="encode_dense")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # add output Dense networks for mu and sigma, units equal to the declared latent_dim.
    mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

    # revise `batch_3.shape` here if you opted not to use 3 Conv2D layers
    return mu, sigma, batch_3.shape

def encoder_model(latent_dim, input_shape):
    """Defines the encoder model with the Sampling layer
    Args:
    latent_dim -- dimensionality of the latent space
    input_shape -- shape of the dataset batch

    Returns:
    model -- the encoder model
    conv_shape -- shape of the features before flattening
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    mu, sigma, conv_shape = encoder_layers(inputs, latent_dim=LATENT_DIM)
    z = Sampling()((mu, sigma))
    model = tf.keras.Model(inputs, outputs=[mu, sigma, z])
    model.summary()
    return model, conv_shape

def decoder_layers(inputs, conv_shape):
    """Defines the decoder layers.
    Args:
    inputs -- output of the encoder
    conv_shape -- shape of the features before flattening
    Returns:
    tensor containing the decoded output
    """
    units = conv_shape[1] * conv_shape[2] * conv_shape[3]
    x = tf.keras.layers.Dense(units, activation = 'relu', name="decode_dense1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    # reshape output using the conv_shape dimensions
    x = tf.keras.layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decode_reshape")(x)
    # upsample the features back to the original dimensions
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name="decode_conv2d_3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid', name="decode_final")(x)

    return x

def decoder_model(latent_dim, conv_shape):
    """Defines the decoder model.
    Args:
    latent_dim -- dimensionality of the latent space
    conv_shape -- shape of the features before flattening
    Returns:
    model -- the decoder model
    """
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    outputs = decoder_layers(inputs, conv_shape)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model

def kl_reconstruction_loss(inputs, outputs, mu, sigma):
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    return tf.reduce_mean(kl_loss) * -0.5


def vae_model(encoder, decoder, input_shape):
    """Defines the VAE model
    Args:
      encoder -- the encoder model
      decoder -- the decoder model
      input_shape -- shape of the dataset batch

    Returns:
      the complete VAE model
    """
    ### START CODE HERE ###
    inputs = tf.keras.layers.Input(shape=input_shape)

    # get mu, sigma, and z from the encoder output
    mu, sigma, z = encoder(inputs)

    # get reconstructed output from the decoder
    reconstructed = decoder(z)

    # define the inputs and outputs of the VAE
    model = tf.keras.Model(inputs=inputs, outputs=reconstructed)

    # add the KL loss
    loss = kl_reconstruction_loss(inputs, z, mu, sigma)
    model.add_loss(loss)

    ### END CODE HERE ###
    return model

def get_models(input_shape, latent_dim):
    encoder, conv_shape = encoder_model(latent_dim=latent_dim, input_shape=input_shape)
    decoder = decoder_model(latent_dim=latent_dim, conv_shape=conv_shape)
    vae = vae_model(encoder, decoder, input_shape=input_shape)
    return encoder, decoder, vae

encoder, decoder, vae = get_models(input_shape=(64,64,3,), latent_dim=LATENT_DIM)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
loss_metric = tf.keras.metrics.Mean()
mse_loss = tf.keras.losses.MeanSquaredError()
bce_loss = tf.keras.losses.BinaryCrossentropy()

# Training loop. Display generated images each epoch

### START CODE HERE ###
epochs = 70
### END CODE HERE ###

random_vector_for_generation = tf.random.normal(shape=[16, LATENT_DIM])

for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(training_dataset):
        with tf.GradientTape() as tape:
            ### START CODE HERE ###
            reconstructed = vae(x_batch_train)

            # compute reconstruction loss
            flattened_inputs = tf.reshape(x_batch_train, shape=[-1])
            flattened_outputs = tf.reshape(reconstructed, shape=[-1])
            loss = mse_loss(flattened_inputs, flattened_outputs) * (64 * 64 * 3)

            # add KLD regularization loss
            loss += sum(vae.losses)

            # get the gradients and update the weights
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            # compute the loss metric
            loss_metric(loss)

            if step % 100 == 0:
                print('Epoch: %s step: %s mean loss = %s' % (epoch, step, loss_metric.result().numpy()))

vae.save('anime.h5')