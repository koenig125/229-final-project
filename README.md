# 229-final-project
Source code for CS229 final project, Stanford University.

## Feature Extraction

We utilize ResNet-50, ResNet-101, and ResNet-152 models to generate three separate sets of embeddings for our dataset. Before passing our images through these models, we perform some minimal normalization. We extract our embeddings from the models before passing through their final fully connected layers which map to the 1,000 image classes in ImageNet. To run our script:

```python generate_embeddings.py --batch_size 128 --res [r50, r101, r152] --out dir /images```

## Training

We use scikit-learn implementations of Support Vector Machines (SVM) and Multi-Layer Perceptron (MLP).

### SVM

Train a SVM using ResNet50 embeddings with rbf kernel and a C value of 10 (see sklearn docs):

```python svm.py -f r50 -k rbf -c 1.0 -g scale```

### MLP

Train a MLP for a maximum of 1000 epochs with 1 hidden layer of dimension 500:

```python mlp.py -f r50 -e 1000 -l 500 -lr 0.001 -a 0.0001```

### Saving Models

Models can be saved after training by using the -s flag. This is useful if it takes a long time to train.

```python svm.py -f r50 -s```

### Defining Number Training Examples

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

Saved SVM and MLP models appear in the `models` directory. They can be loaded and evaluated against the validation or test set using the `evaluate.py` script. This script also allows for creating a confusion matrix from a model's predictions.

Evaluate on validation set:

```python evaluate.py -f r50 -m svm_r50_rbf_1.0_scale_5000```

Evaluate on test set:

```python evaluate.py -f r50 -m svm_r50_rbf_1.0_scale_5000 -t```

Create confusion matrix:

```python evaluate.py -f r50 -m svm_r50_rbf_1.0_scale_5000 -c```
