"""
Script to run SVM classification on image embeddings from the LeafSnap dataset.
"""

import argparse
import time

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import utils

start_time = time.time()


def parse_args():
    """
    Parses the svm arguments.
    """
    parser = argparse.ArgumentParser(description="SVM args.")

    # Should be 'r50', 'r101', 'r152', or 'overfeat'
    parser.add_argument('-f', '--feature_extractor', type=str, required=True,
                        help='Feature extract used to produce image embeddings')
    parser.add_argument('-s', '--hyperparameter_search', default=False, action='store_true',
                        help='Conduct hyperparameter search for best model')
    parser.add_argument('-k', '--kernel', type=str, default='rbf',
                        help='SVM kernel type')
    parser.add_argument('-c', '--regularization', type=float, default=1.0,
                        help='SVM regularization parameter')
    parser.add_argument('-g', '--gamma', type=str, default='scale',
                        help='SVM gamma parameter')
    parser.add_argument('-n', '--num_examples', type=int, default=1000000,
                        help='Number training examples')

    return parser.parse_args()


def hyperparameter_search(svc, X, y):
    """
    Search hyperparameter space for best cross-validation score on data.

    :param - svc: SVC model type
    :param - X: 2D numpy array holding the training image embeddings
    :param - y: 1D numpy array holding the training image labels
    return: Optimal model found from exhaustive search over parameter grid.
    """
    parameters = [{
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10, 100], 
        'gamma': ['scale'],
    }]
    clf = GridSearchCV(svc, parameters, iid=False, cv=3)
    clf.fit(X, y)
    return clf.best_estimator_


def main(args):
    # Load embeddings and labels.
    print('Loading data...')
    embeddings = utils.load_embeddings(args.feature_extractor)
    labels = utils.load_labels(args.feature_extractor)

    # Generate training/validation/testing splits
    print('Splitting data...')
    X_train, X_temp, y_train, y_temp = utils.generate_splits(embeddings, labels, test_size=0.2)
    X_val, X_test, y_val, y_test = utils.generate_splits(X_temp, y_temp, test_size=0.5)
    X_train, X_val, X_test = utils.normalize_data(X_train, X_val, X_test)

    # Train classifier and make predictions.
    if args.hyperparameter_search:
        print('Conducting hyperparameter search...')
        svc = hyperparameter_search(SVC(), X_train[:args.num_examples], y_train[:args.num_examples])
    else:
        print('Training model...')
        print('Num examples used:', args.num_examples)
        svc = SVC(C=args.regularization, kernel=args.kernel, gamma=args.gamma)
        svc.fit(X_train[:args.num_examples], y_train[:args.num_examples])

    # Report results.
    print('Making predictions...')
    predictions = svc.predict(X_val)
    utils.accuracy(predictions, y_val)

    # Save model.
    print('Saving model...')
    params = svc.get_params()
    print('Kernel:', params['kernel'], 'C:', str(params['C']), 'Gamma:', params['gamma'])
    model_name = '_'.join([args.feature_extractor, params['kernel'], str(params['C']), params['gamma'], str(args.num_examples)])
    utils.save_model(svc, model_name)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__=='__main__':
	args = parse_args()
	main(args)
