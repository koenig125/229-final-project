"""
Script to explore hyperparameter space for SVM training.
"""

import argparse
import copy

from svm import train, save


def arg_parse():
    parser = argparse.ArgumentParser(description='SVM arguments.')

    # Should be 'r50', 'r101', 'r152', or 'overfeat'
    parser.add_argument('-f', '--feature_extractor', nargs='+', type=str, required=True,
                        help='Feature extractor used to produce image embeddings')
    parser.add_argument('-k', '--kernel', nargs='+', type=str, required=True,
                        help='SVM kernel type')
    parser.add_argument('-c', '--regularization', nargs='+', type=float, required=True,
                        help='SVM regularization parameter')
    parser.add_argument('-g', '--gamma', nargs='+', type=str, required=True,
                        help='SVM gamma parameter')
    parser.add_argument('-n', '--num_examples', nargs='+', type=int, required=True,
                        help='Number training examples')
    parser.add_argument('-s', '--save', default=False, action="store_true",
                        help='Save the trained SVM models.')

    return parser.parse_args()


def hyperparameter_search(args):
    for f in args.feature_extractor:
        for k in args.kernel:
            for r in args.regularization:
                for g in args.gamma:
                    for n in args.num_examples:
                        print('Training model with params:', [f, k, r, g, n])
                        new_args = copy.deepcopy(args)
                        new_args.feature_extractor = f
                        new_args.kernel = k
                        new_args.regularization = r
                        new_args.gamma = g
                        new_args.num_examples = n
                        svc = train(new_args)
                        save(svc, new_args)


if __name__ == '__main__':
    args = arg_parse()
    hyperparameter_search(args)
