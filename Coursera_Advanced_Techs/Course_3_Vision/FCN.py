import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

class_names = ['sky', 'building','column/pole', 'road', 'side walk',
               'vegetation', 'traffic light', 'fence', 'vehicle',
               'pedestrian', 'byciclist', 'void']


def map_filename_to_image_and_mask(t_filename, a_filename, height=224, width=224):

    img_raw = tf.io.read_file(t_filename)
    anno_raw = tf.io.read_file(a_filename)
    image = tf.image.decode_jpeg(img_raw)
    annotation = tf.image.decode_jpeg(anno_raw)

    image = tf.image.resize(image, (height, width,))
    annotation = tf.image.resize(annotation, (height, width,))
    image = tf.reshape(image, (height, width, 3,))
    annotation = tf.cast(annotation, dtype=tf.int32)
    annotation = tf.reshape(annotation, (height, width, 1,))
    stack_list = []

    # Reshape segmentation masks
    for c in range(len(class_names)):
        mask = tf.equal(annotation[:, :, 0], tf.constant(c))
        stack_list.append(tf.cast(mask, dtype=tf.int32))

    annotation = tf.stack(stack_list, axis=2)

    # Normalize pixels in the input image
    image = image / 127.5
    image -= 1

    return image, annotation


# Utilities for preparing the datasets

BATCH_SIZE = 8


def get_dataset_slice_paths(image_dir, label_map_dir):
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(label_map_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_list]

    return image_paths, label_map_paths


def get_training_dataset(image_paths, label_map_paths):
    training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    training_dataset = training_dataset.map(map_filename_to_image_and_mask)
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)

    return training_dataset


def get_validation_dataset(image_paths, label_map_paths):
    validation_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_map_paths))
    validation_dataset = validation_dataset.map(map_filename_to_image_and_mask)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.repeat()

    return validation_dataset


# get the paths to the images
training_image_paths, training_label_map_paths = get_dataset_slice_paths('dataset1/images_prepped_train/','dataset1/annotations_prepped_train/')
validation_image_paths, validation_label_map_paths = get_dataset_slice_paths('dataset1/images_prepped_test/','dataset1/annotations_prepped_test/')

# generate the train and val sets
training_dataset = get_training_dataset(training_image_paths, training_label_map_paths)
validation_dataset = get_validation_dataset(validation_image_paths, validation_label_map_paths)

def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    for i in range(n_convs):
        x = layers.Conv2D(filters=filters,
                          kernel_size=kernel_size,
                          activation=activation,
                          padding='same',
                          name="{}_conv{}".format(block_name, i + 1))(x)

    x = layers.MaxPooling2D(pool_size=pool_size,
                            strides=pool_stride,
                            name="{}_pool{}".format(block_name, i + 1))(x)

    return x

def VGG_16(image_input):
    x = block(image_input, n_convs=2, filters=64, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2), block_name='block1')

    p1 = x

    x = block(x, n_convs=2, filters=128, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2), block_name='block2')

    p2 = x

    x = block(x, n_convs=3, filters=256, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2), block_name='block3')

    p3 = x

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2), block_name='block4')

    p4 = x

    x = block(x, n_convs=3, filters=512, kernel_size=(3, 3), activation='relu',
              pool_size=(2, 2), pool_stride=(2, 2), block_name='block5')

    p5 = x

    vgg = tf.keras.Model(image_input, p5)

    vgg.load_weights(vgg_weights_path)

    n = 4096

    c6 = layers.Conv2D(n,
                       (7, 7),
                       activation='relu',
                       padding='same',
                       name='conv6')(p5)

    c7 = layers.Conv2D(n,
                       (1, 1),
                       activation='relu',
                       padding='same',
                       name='conv7')(c6)

    return (p1, p2, p3, p4, c7)

def fcn8_decoder(convs, n_classes):
    f1, f2, f3, f4, f5, = convs

    o = layers.Conv2DTranspose(n_classes,
                               kernel_size=(4, 4),
                               strides=(2, 2),
                               use_bias=False)(f5)

    o = layers.Cropping2D(cropping=(1, 1))(o)

    o2 = f4

    o2 = (layers.Conv2D(n_classes,
                        (1, 1),
                        activation='relu',
                        padding='same'))(o2)

    o = layers.Add()([o, o2])

    o = (layers.Conv2DTranspose(n_classes,
                                kernel_size=(4, 4),
                                strides=(2, 2),
                                use_bias=False))(o)

    o = layers.Cropping2D(cropping=(1, 1))(o)

    o2 = (layers.Conv2D(n_classes,
                        (1, 1),
                        activation='relu',
                        padding='same'))(f3)

    o = layers.Add()([o, o2])

    o = layers.Conv2DTranspose(n_classes,
                               kernel_size=(8, 8),
                               strides=(8, 8),
                               use_bias=False)(o)

    o = layers.Activation('softmax')(o)

    return o

def segmenation_model():
    inputs = layers.Input(shape=(224, 224, 3,))
    convs = VGG_16(image_input=inputs)
    outputs = fcn8_decoder(convs, 12)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = segmenation_model()

def class_wise_metrics(y_true, y_pred):
    class_wise_iou = []
    class_wise_dice = []
    smoothing_factor = 0.00001

    for i in range(12):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area
        union_area = combined_area - intersection
        iou = (intersection + smoothing_factor) / (union_area + smoothing_factor)
        dice = 2 * ((intersection + smoothing_factor) / (combined_area + smoothing_factor))
        class_wise_iou.append(iou)
        class_wise_dice.append(dice)

    return class_wise_iou, class_wise_dice

sgd = tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

train_count = 367

validation_count = 101

EPOCHS = 15

steps_per_epoch = train_count//BATCH_SIZE
validation_steps = validation_count//BATCH_SIZE

history = model.fit(training_dataset,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_dataset,
                    validation_steps=validation_steps,
                    epochs=EPOCHS)
