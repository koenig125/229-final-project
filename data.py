"""
Exports functions to load image embedding data and prep for use in training.
"""

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

embeddings_dir = 'embeddings/'
labels_dir = 'labels/'


def load_data(feature_extractor):
    """
    Load image embeddings and labels and split them into train/validation/test sets.
    The embedding data is normalized to zero mean and unit variance before returning.

    :param - feature_extractor: Feature extractor used to produce image embeddings
    return: (X_train, X_val, X_test, y_train, y_val, y_test) 6-tuple
    """
    embeddings = load_embeddings(feature_extractor)
    labels = load_labels(feature_extractor)
    X_train, X_temp, y_train, y_temp = generate_splits(embeddings, labels, test_size=0.2)
    X_val, X_test, y_val, y_test = generate_splits(X_temp, y_temp, test_size=0.5)
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


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
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_val_transformed = scaler.transform(X_val)
    X_test_transformed = scaler.transform(X_test)
    return X_train_transformed, X_val_transformed, X_test_transformed
