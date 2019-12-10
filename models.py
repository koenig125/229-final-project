"""
Exports functions for working with sklearn models.
"""

import time

from joblib import dump, load
from sklearn.metrics import accuracy_score

import data
import utils

models_dir = 'models/'


def train_model(model, feature_extractor, num_examples):
    """
    Trains a sklearn model.

    :param - model: initialized sklearn model
    :param - feature extractor: feature extractor used to produce image embeddings
    return: sklearn model fitted to the embeddings produced by feature extractor
    """
    print('Loading data...')
    X_train, X_val, X_test, y_train, y_val, y_test = data.load_data(feature_extractor)

    print('Training model...')
    start_time = time.time()
    model.fit(X_train[:num_examples], y_train[:num_examples])
    print("--- trained model in %s seconds ---" % (time.time() - start_time))

    print('Making predictions...')
    predictions = model.predict(X_val)
    accuracy(predictions, y_val)


def save_model(model, name):
    """
    Saves a trained sklearn model.

    :param - model: trained sklearn model
    :param - name: filename of the model to save
    """
    utils.make_dir(models_dir)
    filename = models_dir + name + '.joblib'
    dump(model, filename)


def load_model(name):
    """
    Loads a saved sklearn model.

    :param - name: filename of the model to load
    return: sklearn model saved at name specified
    """
    filename = models_dir + name + '.joblib'
    return load(filename)


def accuracy(predictions, labels):
    """
    Calculate classification accuracy given predictions and labels.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    """
    accuracy = accuracy_score(labels, predictions)
    print('Accuracy: {0:.5f}'.format(accuracy))
