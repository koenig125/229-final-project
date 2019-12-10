"""
Script to train SVM classifier on image embeddings from the LeafSnap dataset.
"""

import argparse
import time

from sklearn.svm import SVC

import data
import utils


def parse_args():
    """
    Parses the svm arguments.
    """
    parser = argparse.ArgumentParser(description="SVM args.")

    # Should be 'r50', 'r101', 'r152', or 'overfeat'
    parser.add_argument('-f', '--feature_extractor', type=str, required=True,
                        help='Feature extractor used to produce image embeddings')
    parser.add_argument('-k', '--kernel', type=str, default='rbf',
                        help='SVM kernel type')
    parser.add_argument('-c', '--regularization', type=float, default=1.0,
                        help='SVM regularization parameter')
    parser.add_argument('-g', '--gamma', type=str, default='scale',
                        help='SVM gamma parameter')
    parser.add_argument('-n', '--num_examples', type=int, default=1000000,
                        help='Number training examples')

    return parser.parse_args()


def train_and_save(args):
    start_time = time.time()

    print('Feature Extractor:', args.feature_extractor, 'Kernel:', args.kernel, 
    'C:', args.regularization, 'Gamma:', args.gamma, 'N:', args.num_examples)

    print('Loading data...')
    X_train, X_val, X_test, y_train, y_val, y_test = data.load_data(args.feature_extractor)
    
    print('Training model...')
    svc = SVC(C=args.regularization, kernel=args.kernel, gamma=args.gamma)
    svc.fit(X_train[:args.num_examples], y_train[:args.num_examples])

    print('Making predictions...')
    predictions = svc.predict(X_val)
    utils.accuracy(predictions, y_val)

    print('Saving model...')
    model_name = '_'.join([args.feature_extractor, args.kernel, str(args.regularization), args.gamma, str(args.num_examples)])
    utils.save_model(svc, model_name)

    print("--- trained SVM in %s seconds ---" % (time.time() - start_time))


if __name__=='__main__':
	args = parse_args()
	train_and_save(args)
