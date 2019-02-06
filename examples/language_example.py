import tensorflow as tf
from taskdl.tf_task import LifeomicBinaryClassification, ModelRunner
import numpy as np
import subprocess


class BinaryClassification(LifeomicBinaryClassification):

    def __init__(self, serving_shape=None, save_dir='language_model'):
        super().__init__(self, serving_shape, save_dir=save_dir)

    @staticmethod
    def build_graph(x_dict):
        x = x_dict['item']
        embedding = tf.keras.layers.Embedding(20000, 128, input_length=100)(x)
        conv1 = tf.keras.layers.Conv1D(256, 3, padding='valid', activation='relu', strides=1)(embedding)
        conv2 = tf.keras.layers.Conv1D(64,3, padding='valid', activation='relu', strides=1)(conv1)
        flattened = tf.keras.layers.Flatten()(conv2)
        layer1 = tf.keras.layers.Dense(250, activation='relu')(flattened)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(layer1)
        return out

    def build_model(self, features, labels, mode):
        logits = BinaryClassification.build_graph(features)
        z = tf.squeeze(tf.round(logits))

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                "serving_default": tf.estimator.export.PredictOutput(z)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=z, export_outputs=export_outputs)

        labels_expand = tf.expand_dims(labels, 1)
        loss = tf.losses.mean_squared_error(labels_expand, logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0008)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        metrics = self.build_metrics(tf.squeeze(z), labels)
        log_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10)

        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=metrics,
            predictions=logits,
            training_hooks=[log_hook],
            loss=loss,
            train_op=train_op
        )
        return estim_specs

    def load_imdb(self, path, num_words=None, skip_top=0,
                        start_char=1, oov_char=2, index_from=3):
        with np.load(path) as f:
            x_train, labels_train = f['x_train'], f['y_train']
            x_test, labels_test = f['x_test'], f['y_test']

        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        xs = np.concatenate([x_train, x_test])
        labels = np.concatenate([labels_train, labels_test])

        xs = [[start_char] + [w + index_from for w in x] for x in xs]

        if not num_words:
            num_words = max([max(x) for x in xs])

        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
        x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

        return (x_train, y_train), (x_test, y_test)

    def train_test_data_loader(self, train_test_split=0.25):
        (x_train, y_train), (x_test, y_test) = self.load_imdb(path='examples/imdb.npz', num_words=20000)
        sequence = tf.keras.preprocessing.sequence
        x_train = sequence.pad_sequences(x_train, maxlen=100)
        x_test = sequence.pad_sequences(x_test, maxlen=100)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.serving_shape = [None, x_train.shape[1]]
        return {"item": x_train}, y_train, {"item": x_test}, y_test


if __name__ == '__main__':
    model = ModelRunner(BinaryClassification())
    model.run_all(epochs=1000, batch_size=128)
    result = subprocess.check_output(['ls'])



