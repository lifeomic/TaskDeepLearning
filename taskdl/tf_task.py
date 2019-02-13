import tensorflow as tf
from tensorflow.contrib import predictor
from abc import abstractmethod
import numpy as np
import os
import matplotlib.pyplot as plt
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc
from uuid import uuid4
import csv


tf.logging.set_verbosity(tf.logging.INFO)


class ModelRunner(object):

    def __init__(self, model):
        self.model = model

    def __zipdir(self, path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    def save_model(self):
        self.model.save_model(self.model.save_dir + '/model')
        self.model.plot_train_metrics(self.model.save_dir)

    def run_predictions(self, feature_dict):
        return self.model.predict(feature_dict)

    def __run_classification_cohorts(self, output_path):
        cohort_dict, patients, cohort_obj = self.model.run_cohorts()
        predictions = self.run_predictions(cohort_dict)
        predictions = predictions if len(predictions.shape) == 1 else np.argmax(predictions, axis=1)

        cohort_tuples = []

        for i, prediction in enumerate(predictions):
            if str(prediction) in cohort_obj:
                cohort_tuples.append([patients[i], cohort_obj[str(prediction)]])

        with open(output_path, mode='w') as cohort_file:
            writer = csv.writer(cohort_file, delimiter=',', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['patientId', 'cohortName'])
            for cohort_tuple in cohort_tuples:
                writer.writerow(cohort_tuple)

    def run_all(self, epochs=1000, batch_size=256, zip_directory=True, output_cohort='cohort.csv'):
        train_dict, train_labels, test_dict, test_labels = self.model.load_data()
        self.model.train_model(train_dict, train_labels, epochs=epochs, batch_size=batch_size)
        self.save_model()

        if self.model.model_type == 'classification':
            train_predictions = self.run_predictions(train_dict)
            self.model.save_metrics(train_predictions, train_labels, self.model.save_dir, is_train=True)
            if test_dict:
                test_predictions = self.run_predictions(test_dict)
                self.model.save_metrics(test_predictions, test_labels, self.model.save_dir, is_train=False)

            try:
                self.__run_classification_cohorts(output_cohort)
            except NotImplementedError as e:
                print("COHORT NOT IMPLEMENTED")

        if zip_directory:
            zipf = zipfile.ZipFile(self.model.save_dir + '.zip', 'w', zipfile.ZIP_DEFLATED)
            self.__zipdir(self.model.save_dir, zipf)
            zipf.close()


class LifeomicTensorflow(object):

    def __init__(self, serving_shape=None, predict_func=None, save_dir='model_data'):
        self.predict_func = predict_func
        self.serving_shape = serving_shape
        self.model_dir = 'temp_model_' + str(uuid4())
        self.save_dir = save_dir
        self.model = tf.estimator.Estimator(self.build_model, model_dir=self.model_dir)

    @property
    @abstractmethod
    def available_metrics(self):
        raise NotImplementedError("Available metrics must be implemented")

    @property
    @abstractmethod
    def model_type(self):
        raise NotImplementedError("Model Type must be implemented")

    @abstractmethod
    def build_model(self, features, labels, mode):
        raise NotImplementedError("Need to implement building")

    @abstractmethod
    def train_data_loader(self):
        raise NotImplementedError("Data loader needs to be supplied")

    @abstractmethod
    def test_data_loader(self):
        return self.train_data_loader()

    @abstractmethod
    def train_test_data_loader(self, test_split=0.2):
        raise NotImplementedError("Test Data loader must be implemented")

    @abstractmethod
    def plot_train_metrics(self, directory):
        raise NotImplementedError("Plot Metrics must be implemented")

    @abstractmethod
    def load_data(self, train_test_split):
        raise NotImplementedError("Load Data must be implemented")

    @abstractmethod
    def run_cohorts(self):
        raise NotImplementedError("Cohorts is not loaded")

    def train_model(self, feed_dict, labels=None, epochs=100, batch_size=500, shuffle=True):
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=feed_dict, y=labels, batch_size=batch_size, num_epochs=None, shuffle=shuffle
        )
        self.model.train(input_fn, max_steps=epochs)

    def save_model(self, path):
        self.model.export_savedmodel(export_dir_base=path, serving_input_receiver_fn=self.serving_input_fn)

    def get_model(self):
        return self.model

    def predict(self, item_dict):
        if self.predict_func:
            return self.predict_func(item_dict)['output']
        return self.__full_predictions(item_dict)

    @staticmethod
    def load_model(path):
        predict_fn = predictor.from_saved_model(path)
        return LifeomicTensorflow(predict_func=predict_fn)

    def get_train_metric_summaries(self):
        event_file = None
        for filename in os.listdir(self.model_dir):
            if 'tfevents' in filename:
                event_file = filename
                break

        values = {}
        for e in tf.train.summary_iterator(self.model_dir + '/' + event_file):
            for v in e.summary.value:
                if v.tag in self.available_metrics:
                    if v.tag in values:
                        values[v.tag].append(v.simple_value)
                    else:
                        values[v.tag] = [v.simple_value]
        return values

    def get_numpy_split(self, features, labels, test_split=0.25):
        return train_test_split(features, labels, test_size=test_split, random_state=12345)

    def serving_input_fn(self):
        tensor = {'item': tf.placeholder(tf.float32, shape=self.serving_shape)}
        return tf.estimator.export.ServingInputReceiver(tensor, tensor)

    def __full_predictions(self, x):
        preds = self.__predictions(x)
        all_d = []
        for pred in preds:
            all_d.append(pred)
        return np.asarray(all_d)

    def __predictions(self, item_dict):
        input_fn = tf.estimator.inputs.numpy_input_fn(x=item_dict, shuffle=False)
        return self.model.predict(input_fn)


class LifeomicClassification(LifeomicTensorflow):

    model_type = 'classification'

    def __init__(self, serving_shape=None, predict_func=None, save_dir='model_data'):
        super().__init__(serving_shape, predict_func, save_dir)

    def load_data(self, train_test_split=None):
        train_dict, train_labels, test_dict, test_labels = None, None, None, None
        try:
            train_dict, train_labels, test_dict, test_labels = self.train_test_data_loader(train_test_split)
        except NotImplementedError as ne:
            pass

        if train_dict is None:
            train_dict, train_labels = self.train_data_loader()

        if test_dict is None:
            test_dict, test_labels = self.test_data_loader()

        return train_dict, train_labels, test_dict, test_labels

    @abstractmethod
    def load_cohorts(self):
        raise NotImplementedError("Cohorts is not loaded")

    def run_cohorts(self):
        features, patients, cohort_obj = self.load_cohorts()
        if not cohort_obj or not 'classificationCohortMapping' in cohort_obj:
            raise RuntimeError("classificationCohortMapping must be supplied for cohort object")
        return features, patients, cohort_obj['classificationCohortMapping']

    @abstractmethod
    def build_metrics(self, predictions, labels):
        raise NotImplementedError("Building metrics needs implemented")

    @abstractmethod
    def save_metrics(self, predictions, labels, directory, is_train, classes=None):
        raise NotImplementedError("Must contain save metrics")

    def one_hot(self, items, n_classes):
        to_ret = np.zeros((len(items), n_classes))
        for i, label in enumerate(items):
            start = np.zeros(n_classes)
            start[int(label)] = 1.0
            to_ret[i] = start
        return to_ret

    def plot_train_metrics(self, directory):
        summaries = self.get_train_metric_summaries()
        for key, value in summaries.items():
            item_length = len(value)
            x_axis = [i * 100 for i in range(item_length)]
            plt.xlabel("Iterations")
            plt.ylabel(key)
            plt.title("Training %s over time." % key)
            plt.plot(x_axis, value, linewidth=2)
            plt.savefig('%s/%s_plot.png' % (directory, key))
            plt.clf()

    def plot_confusion_matrix(self, predictions, labels, title, path,
                                  classes=None, normalize=False, cmap=plt.cm.Blues):
        cnf_matrix = confusion_matrix(labels, predictions)
        classes = classes if classes else [label for label in range(labels.max() + 1)]

        if normalize:
            cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(path + '.png')
        plt.clf()

    def plot_auc(self, predictions, labels, path_base, classes=None):
        classes = classes if classes else [label for label in range(labels.max() + 1)]
        labels = self.one_hot(labels, len(classes))
        predictions = self.one_hot(predictions, len(classes))

        false_positive = dict()
        true_positive = dict()
        roc_auc = dict()
        for i in range(len(classes)):
            false_positive[i], true_positive[i], _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc[i] = auc(false_positive[i], true_positive[i])

        for i in range(len(classes)):
            plt.figure()
            plt.plot(false_positive[i], true_positive[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Class %s ROC Curve' % classes[i])
            plt.legend(loc="lower right")
            plt.savefig('%s-%s.png' % (path_base, classes[i]))
            plt.clf()


class LifeomicMultiClassification(LifeomicClassification):

    available_metrics = ['loss', 'accuracy']

    def __init__(self, serving_shape=None, predict_func=None, save_dir='model_data'):
        super().__init__(serving_shape, predict_func, save_dir)

    def build_metrics(self, predictions, labels):
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        return metrics

    def save_metrics(self, predictions, labels, directory, is_train, classes=None):
        data_type = 'Train' if is_train else 'Test'
        labels = labels if len(labels.shape) == 1 else np.argmax(labels, axis=1)
        predictions = predictions if len(predictions.shape) == 1 else np.argmax(predictions, axis=1)

        self.plot_confusion_matrix(predictions, labels,
                                   '%s Confusion Matrix' % data_type,
                                   '%s/%s-ConfusionMatrix' % (directory, data_type),
                                   classes)
        self.plot_auc(predictions, labels, '%s/%s_ROC' % (directory, data_type), classes)


class LifeomicBinaryClassification(LifeomicClassification):

    available_metrics = ['loss', 'accuracy', 'auc', 'recall', 'precision']

    def build_metrics(self, predictions, labels, save_dir=''):
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        return metrics

    def save_metrics(self, predictions, labels, directory, is_train, classes=None):
        data_type = 'Train' if is_train else 'Test'
        self.plot_confusion_matrix(predictions, labels,
                                   '%s Confusion Matrix' % data_type,
                                   '%s/%s-ConfusionMatrix' % (directory, data_type),
                                   classes)
        self.plot_auc(predictions, labels, '%s/%s_ROC' % (directory, data_type), classes)


