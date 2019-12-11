"""
Script to evaluate saved sklearn models that have been trained on the LeafSnap image embeddings.
"""

import argparse

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
    parser.add_argument('-n', '--top_n', type=int, default=5,
                        help='N for top-N accuracy')
    parser.add_argument('-t', '--is_test', default=False, action="store_true",
                        help='Evaluate on test split.')
    parser.add_argument('-c', '--confusion_matrix', default=False, action="store_true",
                        help='Create confusion matrix from predictions.')

    return parser.parse_args()


def eval_saved_model(feature_extractor, model_name, n, is_test, cm=False):
    print('Loading data...')
    X_train, X_val, X_test, y_train, y_val, y_test = data.load_data(args.feature_extractor)
    X, y = (X_test, y_test) if is_test else (X_val, y_val)
        
    print('Loading model...')
    model = models.load_model(model_name)
    model_type = model_name[:3] # will be svm or mlp
    predictions = models.top_1_accuracy(model, X, y)
    models.top_n_accuracy(model, X, y, n, model_type)

    if cm:
        print('Creating confusion matrix...')
        cm_path = 'images/cm_' + model_name + '.png'
        utils.make_cm(predictions, y, cm_path)


def main(args):
    eval_saved_model(args.feature_extractor, args.model_name, args.top_n, 
                    args.is_test, args.confusion_matrix)


if __name__ == '__main__':
    args = parse_args()
    main(args)
