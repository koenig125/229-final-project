import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from joblib import dump, load
from matplotlib import cm

# Mapping from labels to species for the LeafSnap image embeddings
labels_to_species = {
    # TODO: Fill in once mapping of label numbers to species names is known
}

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


def accuracy(predictions, labels):
    """
    Calculate classification accuracy given predictions and labels.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    """
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    print('Accuracy: {0:.5f}'.format(accuracy))


def confusion_matrix(predictions, labels, title, path):
    """
    Calculate confusion matrix given predictions and labels.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    """
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    plot_heatmap(confusion_matrix, title, path, 'Predicted Label', 'True Label')


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
