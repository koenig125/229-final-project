"""
Exports functions for working with sklearn models.
"""

import time

import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score

import data
import utils

models_dir = 'models/'


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

    top_1_accuracy(model, X_val, y_val)


def top_1_accuracy(model, X, y, exclude_mislabeled=False):
    """
    Calculate classification top_1_accuracy.

    :param - model: trained sklearn model
    :param - X: image embedding data
    :param - y: labels for embeddings
    :param - exclude_mislabeled: exclude 
    mislabeled images from accuracy calculations
    return: 1D numpy array of predictions
    """
    print('Predicting top-1...')
    predictions = model.predict(X)

    if exclude_mislabeled:
        idxs = utils.get_valid_indices(y)
        y, predictions = y[idxs], predictions[idxs]

    top_1_accuracy = accuracy_score(y, predictions)
    print('Top_1_accuracy: {0:.5f}'.format(top_1_accuracy))
    return predictions


def top_n_accuracy(model, X, y, n, model_type, exclude_mislabeled=False):
    """
    Calculates the top-n accuracy.

    :param - model: trained sklearn model
    :param - X: image embedding data
    :param - y: labels for embeddings
    :param - n: n for top-n accuracy
    :param - model_type: class of sklearn model, ie SVM, MLP, etc.
    :param - exclude_mislabeled: exclude mislabeled images from accuracy calculations
    return: 2D numpy array of top-n predictions of size (num_samples, n)
    """
    print('Predicting top-n...')
    if model_type.lower() == 'svm':
        class_scores = model.decision_function(X)
    else:
        class_scores = model.predict_proba(X)
    predictions = np.argsort(class_scores, axis=1)[:, -n:]

    if exclude_mislabeled:
        idxs = utils.get_valid_indices(y)
        y, predictions = y[idxs], predictions[idxs]
        
    matches = [1 if y[i] in predictions[i] else 0 for i in range(len(y))]
    top_n_accuracy = sum(matches) / len(matches)
    print('Top-' + str(n) + ' accuracy:' + str(top_n_accuracy))
    return predictions
