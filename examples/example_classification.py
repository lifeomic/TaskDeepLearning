import tensorflow as tf
from taskdl.tf_task import LifeomicMultiClassification, ModelRunner
import numpy as np
import pandas as pd


class MnistClassification(LifeomicMultiClassification):

    def __init__(self, serving_shape=None):
        LifeomicMultiClassification.__init__(self, serving_shape)

    @staticmethod
    def build_graph(x_dict):
        x = x_dict['item']
        layer1 = tf.layers.dense(x, 256, activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_uniform_initializer())
        layer2 = tf.layers.dense(layer1, 256, activation=tf.nn.relu,
                                 kernel_initializer=tf.glorot_uniform_initializer())
        out = tf.layers.dense(layer2, 10,
                              kernel_initializer=tf.glorot_uniform_initializer())
        return out

    def build_model(self, features, labels, mode):
        logits = MnistClassification.build_graph(features)
        z = tf.argmax(logits, 1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                "serving_default": tf.estimator.export.PredictOutput(z)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=z, export_outputs=export_outputs)

        labels_max = tf.argmax(labels, 1)
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0008)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        metrics = self.build_metrics(z, labels_max)
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
        data = pd.read_csv('examples/mnist_train.csv', delimiter=',', header=None).values
        features = data[:, 1:785].astype(np.float32)
        labels = data[:, 0]
        softmax_labels = np.zeros((len(labels), 10))
        for i, label in enumerate(labels):
            start = np.zeros(10)
            start[label] = 1.0
            softmax_labels[i] = start

        self.serving_shape = features.shape
        features_train, features_test, labels_train, labels_test = self.get_numpy_split(features, softmax_labels)
        return {"item": features_train}, labels_train, {"item": features_test}, labels_test


if __name__ == '__main__':
    mr = ModelRunner(MnistClassification())
    mr.run_all(epochs=2000, batch_size=500)