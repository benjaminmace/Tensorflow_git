import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([64, 128, 512, 1024, 2048]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.15, 0.2, 0.25, 0.3, 0.35]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'RMSprop']))

METRIC_ACCURACY = 'accuracy'

log_dir = 'E:\\Code\\Projects\\Education\\TF_Hypertuning\\hparam_tuning'

with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=10)
    _, accuracy = model.evaluate(x_test, y_test)

    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        accuracy_score = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy_score, step=1)


session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in HP_DROPOUT.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1
