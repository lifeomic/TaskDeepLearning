import tensorflow as tf
from taskdl.tf_task import LifeomicBinaryClassification, ModelRunner
import numpy as np


class BinaryClassification(LifeomicBinaryClassification):

    def __init__(self, serving_shape=None):
        super().__init__(self, serving_shape)

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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

    def train_test_data_loader(self, train_test_split=0.25):
        imdb = tf.keras.datasets.imdb
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
        sequence = tf.keras.preprocessing.sequence
        x_train = sequence.pad_sequences(x_train, maxlen=100)
        x_test = sequence.pad_sequences(x_test, maxlen=100)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        self.serving_shape = [None, x_train.shape[1]]
        return {"item": x_train}, y_train, {"item": x_test}, y_test


model = ModelRunner(BinaryClassification())
model.run_all(epochs=1000, batch_size=128)
print("")



