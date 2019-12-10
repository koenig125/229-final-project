"""
Script to train MLP classifier on image embeddings from the LeafSnap dataset.
"""

import argparse

from sklearn.neural_network import MLPClassifier

import models


def parse_args():
    parser = argparse.ArgumentParser(description="MLP args.")

    # Should be 'r50', 'r101', 'r152', or 'overfeat'
    parser.add_argument('-f', '--feature_extractor', type=str, required=True,
                        help='Feature extractor used to produce image embeddings')
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='Max number of epochs to train')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Step size for optimizer updates')
    parser.add_argument('-l', '--layers', nargs='+', type=int, default=(500,),
                        help='Tuple of hidden layer sizes')
    parser.add_argument('-a', '--alpha', type=float, default=0.0001,
                        help='MLP regularization parameter')
    parser.add_argument('-n', '--num_examples', type=int, default=1000000,
                        help='Number training examples')

    return parser.parse_args()


def train(args):
    mlp = MLPClassifier(hidden_layer_sizes=args.layers, alpha=args.alpha, 
    learning_rate_init=args.learning_rate, max_iter=args.epochs)
    models.train_model(mlp, args.feature_extractor, args.num_examples)
    return mlp


def save(mlp, args):
    params = [args.feature_extractor, str(args.learning_rate), 
    str(args.epochs), str(args.alpha), str(args.num_examples)]
    params.extend([str(l) for l in args.layers])
    model_name = 'mlp_' + '_'.join(params)
    models.save_model(mlp, model_name)


def main():
    args = parse_args()
    print('Feature Extractor:', args.feature_extractor, 'Epochs:', args.epochs, 
    'Learning Rate:', args.learning_rate, 'Layers:', args.layers, 'Alpha:', args.alpha)
    mlp = train(args)
    save(mlp, args)


if __name__=='__main__':
    main()
