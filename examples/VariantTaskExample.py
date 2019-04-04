import tensorflow as tf
from taskdl.tf_task import LifeomicBinaryClassification, ModelRunner
import numpy as np
import json
import subprocess

"""
Basic example of public TCGA BRCA somatic variants, predicting triple negative breast cancer 
"""
class VariantTaskExample(LifeomicBinaryClassification):

    def __init__(self, serving_shape=None, save_dir='variant_model'):
        super().__init__(self, serving_shape, save_dir=save_dir)

    @staticmethod
    def build_graph(x_dict, mode):
        x = x_dict['item']
        layer1 = tf.layers.dense(x, 512, activation=tf.nn.relu)
        do1 = tf.layers.dropout(layer1, 0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
        layer2 = tf.layers.dense(do1, 512, activation=tf.nn.relu)
        do1 = tf.layers.dropout(layer2, 0.5, training= mode == tf.estimator.ModeKeys.TRAIN)
        out = tf.layers.dense(do1, 1, activation=tf.nn.sigmoid)
        return out

    def build_model(self, features, labels, mode):
        logits = VariantTaskExample.build_graph(features, mode)
        z = tf.squeeze(tf.round(logits))

        if mode == tf.estimator.ModeKeys.PREDICT:
            z = tf.cast(z, tf.int32)
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

    def train_test_data_loader(self, train_test_split=0.1):
        with open('variant_data.json', 'r') as f:
            loaded_data = json.load(f)
        labels = np.asarray(loaded_data['labels'])
        features = np.asarray(loaded_data['features']).astype(np.float32)
        features_train, features_test, labels_train, labels_test = self.get_numpy_split(features, labels, test_split=0.1)
        self.serving_shape = [None, features_train.shape[1]]
        return {"item": features_train}, labels_train, {"item": features_test}, labels_test

    def load_cohorts(self):
        with open('variant_data.json', 'r') as f:
            loaded_data = json.load(f)
        features = np.asarray(loaded_data['features']).astype(np.float32)
        patients = loaded_data['patients']
        cohort_mapping = {
            'classificationCohortMapping': {
                '0': 'Predicted Not TN',
                '1': 'Predicted TN'
            }
        }
        return {"item": features}, patients, cohort_mapping


if __name__ == '__main__':
    model = ModelRunner(VariantTaskExample())
    model.run_all(epochs=1000, batch_size=128)
    result = subprocess.check_output(['ls'])



