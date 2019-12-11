import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import confusion_matrix

# Mapping from labels to species for the LeafSnap image embeddings
labels_to_species = {
    # TODO: Fill in once mapping of label numbers to species names is known
}

images_dir = 'images/'


def make_cm(predictions, labels, path):
    """
    Calculate confusion matrix given predictions and labels and plot.

    :param - predictions: list for which index i holds the prediction for embedding i
    :param - labels: list for which index i holds ground truth label for embedding i
    """
    cm = confusion_matrix(labels, predictions)
    cm = cm / np.sum(cm, axis=1)[:, None] # Normalize each row to sum to 1
    plot_heatmap(cm, path, 'Predicted Label', 'True Label')


def plot_heatmap(matrix, path, xlabel=None, ylabel=None):
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
    heatmap = sns.heatmap(df_cm)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
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
