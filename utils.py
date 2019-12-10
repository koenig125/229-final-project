import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
from joblib import dump, load
from matplotlib import cm
from sklearn.model_selection import train_test_split

# Mapping from labels to species for the LeafSnap image embeddings
labels_to_species = {
    # TODO: Fill in once mapping of label numbers to species names is known
}

embeddings_dir = 'embeddings/'
labels_dir = 'labels/'
models_dir = 'models/'
images_dir = 'images/'


def save_model(model, name):
    """
    Saves a trained sklearn model.

    :param - model: trained sklearn model
    :param - name: filename of the model to save
    """
    make_dir(models_dir)
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


def load_embeddings(feature_extractor):
    """
    Loads embeddings produced from ResNet or OverFeat feature extractors. 

    :param - filename: filename for embeddings
    return: 2D numpy array of image embeddings
    """
    filename = 'embeddings_' + feature_extractor + '.pt'
    path = embeddings_dir + filename
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = torch.load(path, map_location=torch.device(device))
    X = embeddings.cpu().numpy()
    return X


def load_labels(feature_extractor):
    """
    Loads labels for embeddings produced from ResNet or OverFeat feature extractors. 

    :param - filename: filename for labels
    return: 1D numpy array of image labels
    """
    filename = 'labels_' + feature_extractor + '.pt'
    path = labels_dir + filename
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = torch.load(path, map_location=torch.device(device))
    y = labels.cpu().numpy()
    return y


def generate_splits(X, y, test_size):
    """
    Generate train/test splits for embedding data. Random seed is set to 229 for consistency
    during development and so that the SVC and MLP models are evaluated on the same test set.
    
    :param - X: 2D numpy array holding the image embeddings
    :param - y: 1D numpy array holding the image labels
    :param - test_size: percentage of data to make test
    return: (X_train, X_test, y_train, y_test) tuple
    """
    return train_test_split(X, y, test_size=test_size, random_state=229)


def normalize_data(X_train, X_val, X_test):
    """
    Standardizes X_train to zero mean and unit variance, then applies the 
    same transformation that was executed on X_train to X_val and X_test.

    :param - X_train: 2D numpy array holding training image embeddings
    :param - X_val: 2D numpy array holding validation image embeddings
    :param - X_test: 2D numpy array holding testing image embeddings
    return: Normalized (X_train, X_test) tuple
    """
    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_val_transformed = scaler.transform(X_val)
    X_test_transformed = scaler.transform(X_test)
    return X_train_transformed, X_val_transformed, X_test_transformed


def confusion_matrix(predictions, labels, title, path):
    """
    Calculate confusion matrix given predictions and labels.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    """
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    plot_heatmap(confusion_matrix, title, path, 'Predicted Label', 'True Label')


def accuracy(predictions, labels):
    """
    Calculate classification accuracy given predictions and labels.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    """
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    print('Accuracy: {0:.5f}'.format(accuracy))


def cross_validation_score(classifier, X, y):
    """
    Calculate cross-validation accuracy given a classifier and data set.

    :param - classifier: trained classifier
    :param - X: 2D numpy array holding the image embeddings
    :param - y: 1D numpy array holding the image labels
    return: score for 5-fold cross-validation
    """
    cv_scores = sklearn.model_selection.cross_val_score(classifier, X, y, cv=3)
    print('Mean Cross Validation Accuracy:', sum(cv_scores) / len(cv_scores))


def plot_heatmap(matrix, title, path, xlabel=None, ylabel=None):
    """
    Plots the provided matrix as a heatmap.

    :param - matrix: 2D numpy array representing values to plot in heatmap
    :param - title: title for the plot
    :param - path: save path for plot
    :param - xlabel: label for x-axis
    :param - ylabel: label for y-axis
    """
    plt.close('all')
    df_cm = pd.DataFrame(matrix)
    _ = plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm, annot=True, cmap=sns.cm.rocket_r)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    make_dir(images_dir)
    plt.savefig(path)


def make_dir(dir_name):
    '''
    Creates a directory if it doesn't yet exist.
    
    :param - dir_name: Name of directory to be created.
    '''
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, dir_name + '/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
