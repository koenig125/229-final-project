"""
Script to evaluate saved sklearn models that have been trained on the LeafSnap image embeddings.
"""

import argparse

import numpy as np

import data
import models
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="SVM args.")

    # Should be 'r50', 'r101', 'r152', or 'overfeat'
    parser.add_argument('-f', '--feature_extractor', type=str, required=True,
                        help='Feature extract used to produce image embeddings')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        help='Name of model to evaluate')
    parser.add_argument('-n', '--top_n', type=int,
                        help='N for top-N accuracy')
    parser.add_argument('-t', '--is_test', default=False, action="store_true",
                        help='Evaluate on test split.')
    parser.add_argument('-e', '--errors', default=False, action="store_true",
                        help='Calculate where model makes misclassifications.')                        
    parser.add_argument('-c', '--confusion_matrix', default=False, action="store_true",
                        help='Create confusion matrix from predictions.')
    parser.add_argument('-em', '--exclude_mislabeled', default=False, action="store_true",
                        help='Exclude mislabeled classes from accuracy scores.')
    
    return parser.parse_args()


def eval_saved_model(args):
    print('Loading data...')
    X_train, X_val, X_test, y_train, y_val, y_test = data.load_data(args.feature_extractor)
    X, y = (X_test, y_test) if args.is_test else (X_val, y_val)
        
    print('Loading model...')
    model = models.load_model(args.model_name)
    model_type = args.model_name[:3] # will be svm or mlp
    predictions = models.top_1_accuracy(model, X, y, args.exclude_mislabeled)
    if args.top_n:
        models.top_n_accuracy(model, X, y, args.top_n, model_type, args.exclude_mislabeled)

    if args.errors:
        print('Finding misclassifications...')
        cm = utils.calculate_cm(predictions, y)
        cm = utils.normalize_cm(cm)
        misclassifications(cm)
    if args.confusion_matrix:
        print('Creating confusion matrix...')
        cm_path = 'images/cm_' + args.model_name + '.png'
        utils.plot_cm(predictions, y, cm_path)


def misclassifications(cm):
    species_info = []
    for i in range(cm.shape[0]):
        accuracy = cm[i][i]

        # top 6 predicted labels for species i
        labels = np.argsort(cm[i])[-6:]
        
        # exclude species i itself from the list
        labels = [l for l in labels if l != i][-5:]

        # percentage of misclassifications of species i
        # that are due to predicting species l instead
        percentages = [cm[i][l] for l in labels]

        species_info.append((accuracy, labels, percentages))
    
    print('Leaf Species Classification Accuracies:')
    accuracies = [s[0] for s in species_info]
    idxs = np.argsort(accuracies)
    for idx in idxs:
        print('Accuracy for Species {0}: {1:.2f}'.format(utils.labels_to_species[idx], accuracies[idx]))
        labels = species_info[idx][1]
        percentages = species_info[idx][2]
        print('Misclassified as:')
        for i in range(len(labels)):
            print('Species {0}: {1:.2f}'.format(utils.labels_to_species[labels[i]], percentages[i]))
        print()


def main(args):
    eval_saved_model(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
