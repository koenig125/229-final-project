"""
Script to train SVM classifier on image embeddings from the LeafSnap dataset.
"""

import argparse

from sklearn.svm import SVC

import models


def parse_args():
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
    parser.add_argument('-s', '--save', default=False, action="store_true",
                        help='Save the trained SVM model.')

    return parser.parse_args()


def train(args):
    svc = SVC(C=args.regularization, kernel=args.kernel, gamma=args.gamma)
    models.train_model(svc, args.feature_extractor, args.num_examples)
    return svc


def save(svc, args):
    params = [args.feature_extractor, args.kernel, 
    str(args.regularization), args.gamma, str(args.num_examples)]
    model_name = 'svm_' + '_'.join(params)
    models.save_model(svc, model_name)


def main():
    args = parse_args()
    print('Feature Extractor:', args.feature_extractor, 'Kernel:', args.kernel, 
    'C:', args.regularization, 'Gamma:', args.gamma, 'N:', args.num_examples)
    svc = train(args)
    if args.save:
        save(svc, args)


if __name__=='__main__':
    main()
