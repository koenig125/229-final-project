"""
Script to explore hyperparameter space for MLP training.
"""

import argparse
import copy

from mlp import train, save


def arg_parse():
    parser = argparse.ArgumentParser(description='MLP arguments.')

    # Should be 'r50', 'r101', 'r152', or 'overfeat'
    parser.add_argument('-f', '--feature_extractor', nargs='+', type=str, required=True,
                        help='Feature extractor used to produce image embeddings')
    parser.add_argument('-e', '--epochs', nargs='+', type=int, required=True,
                        help='Max number of epochs to train')
    parser.add_argument('-lr', '--learning_rate', nargs='+', type=float, required=True,
                        help='Step size for optimizer updates')
    parser.add_argument('-l', '--layers', nargs='+', type=int, required=True,
                        help='Tuple of hidden layer sizes')
    parser.add_argument('-a', '--alpha', nargs='+', type=float, required=True,
                        help='MLP regularization parameter')
    parser.add_argument('-n', '--num_examples', nargs='+', type=int, required=True,
                        help='Number training examples')
    parser.add_argument('-s', '--save', default=False, action="store_true",
                        help='Save the trained MLP models.')
    
    return parser.parse_args()


def hyperparameter_search(args):
    l = args.layers
    for f in args.feature_extractor:
        for e in args.epochs:
            for lr in args.learning_rate:
                for a in args.alpha:
                    for n in args.num_examples:
                        print('Training model with params:', [f, e, lr, l, a, n])
                        new_args = copy.deepcopy(args)
                        new_args.feature_extractor = f
                        new_args.epochs = e
                        new_args.learning_rate = lr
                        new_args.layers = l
                        new_args.alpha = a
                        new_args.num_examples = n
                        mlp = train(new_args)
                        save(mlp, new_args)


if __name__ == '__main__':
    args = arg_parse()
    hyperparameter_search(args)
