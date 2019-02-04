import tensorflow as tf
from tensorflow.contrib import predictor
from abc import abstractmethod
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)


class LifeomicTensorflow(object):

    def __init__(self, serving_shape=None, predict_func=None):
        self.predict_func = predict_func
        self.serving_shape = serving_shape
        self.model = tf.estimator.Estimator(self.build_model, model_dir='temp_model')

    @abstractmethod
    def build_model(self, features, labels, mode):
        raise NotImplementedError("Need to implement building")

    @abstractmethod
    def serving_input_fn(self):
        raise NotImplementedError("Need to implement serving")

    def train_model(self, feed_dict, labels=None, epochs=100, batch_size=500, shuffle=True):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=feed_dict, y=labels, batch_size=batch_size, num_epochs=None, shuffle=shuffle
        )
        self.model.train(input_fn, max_steps=epochs)

    def save_model(self, path):
        self.model.export_savedmodel(export_dir_base=path, serving_input_receiver_fn=self.serving_input_fn)

    def get_model(self):
        return self.model

    def __full_predictions(self, x):
        preds = self.__predictions(x)
        all_d = []
        for pred in preds:
            all_d.append(pred)
        return np.asarray(all_d)

    def __predictions(self, item_dict):
        input_fn = tf.estimator.inputs.numpy_input_fn(x=item_dict, shuffle=False)
        return self.model.predict(input_fn)

    def predict(self, item_dict):
        if self.predict_func:
            return self.predict_func(item_dict)['output']
        return self.__full_predictions(item_dict)

    @staticmethod
    def load_model(path):
        predict_fn = predictor.from_saved_model(path)
        return LifeomicTensorflow(predict_func=predict_fn)


class TFMetrics(object):

    def __init__(self):
        pass


class TFBinaryClassificationMetrics(object):

    def __init__(self):
        pass


class TFMultiClassificationMetrics(object):

    def __init__(self):
        pass


class TFRegressionMetrics(object):

    def __init__(self):
        pass


class TFAutoEncoderMetrics(object):

    def __init__(self):
        pass

