



class Callback(object):
    def __init__(self):
        self.validation_data = None
        self.model = None

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of the epoch"""

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch"""

#Tensorboard

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

model.fit(train_batches,
          epochs=10,
          validation_data=validation_batches,
          callbacks=[tensorboard_callback])

#Saving weights with formated string

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=5,
          validation_data=validation_batches,
          verbose=2,
          callbacks=[ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
                     ])

#saving entire model
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=1,
          validation_data=validation_batches,
          verbose=2,
          callbacks=[ModelCheckpoint('saved_model', verbose=1)
                     ])

#reduce LR on plateau
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_batches,
          epochs=50,
          validation_data=validation_batches,
          callbacks=[ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.2, verbose=1,
                                       patience=1, min_lr=0.001),
                     TensorBoard(log_dir='./log_dir')])

#CSV Logger
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

csv_file = 'training.csv'

model.fit(train_batches,
          epochs=5,
          validation_data=validation_batches,
          callbacks=[CSVLogger(csv_file)
                     ])

#LR scheduler
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 1
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


model.fit(train_batches,
          epochs=5,
          validation_data=validation_batches,
          callbacks=[LearningRateScheduler(step_decay, verbose=1),
                     TensorBoard(log_dir='./log_dir')])