# 229-final-project
Source code for CS229 final project, Stanford University.

## Training

We use scikit-learn implementations of Support Vector Machines (SVM) and Multi-Layer Perceptron (MLP).

### SVM

Train a SVM using ResNet50 embeddings with rbf kernel and a C value of 10 (see sklearn docs):

```python svm.py -f r50 -k rbf -c 1.0 -g scale```

### MLP

Train a MLP for a maximum of 1000 epochs with 2 hidden layers of dimension 500 and 250:

```python mlp.py -f r50 -e 1000 -l 500 250 -lr 0.001 -a 0.0001```

### Training Examples

Models can be trained using a subset of the training data for faster results and experimentation.

```python svm.py -f r50 -n 5000   # run with 5000 training examples```

```python mlp.py -f r50 -n 10000   # run with 10000 training examples```

## Hyperparameter Search

Hyperparameter search is implemented as a simple grid search. This search is implemented manually (as opposed to using sklearn's GridSearch module) so that the results of each model trained can be reported and each model can be saved. This is useful as some of the models take a long time to train, so we wouldn't want to have to retrain models for further evaluation.

### SVM

Search SVM hyperparameter space over kernls = ['rbf', 'linear'] and C = [1.0, 10.0]:

```python hyperparameter_search_svm.py -f r50 -k linear rbf -c 1.0 10.0 -g scale -n 1000```

### MLP

Search MLP hyperparameter space over learning rate = ['0.01', '0.001'] and alpha = [0.01, 0.0001]:

```python hyperparameter_search_mlp.py -f r50 -e 1000 -lr 0.01 0.001 -l 500 250 -a 0.01 0.0001 -n 1000```

## Evaluating Saved Models

Trained SVM and MLP models are saved to the `models` directory. They can be loaded and evaluated against the validation or test set using the `evaluate.py` script. This script also allows for creating a confusion matrix from a model's predictions.

Evaluate on validation set:

```python evaluate.py -f r50 -m svm_r50_rbf_1.0_scale_5000```

Evaluate on test set:

```python evaluate.py -f r50 -m svm_r50_rbf_1.0_scale_5000 -t```

Create confusion matrix:

```python evaluate.py -f r50 -m svm_r50_rbf_1.0_scale_5000 -c```
