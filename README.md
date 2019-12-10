# 229-final-project
Source code for CS229 final project, Stanford University.

## SVM Training
Train an SVM using embeddings from ResNet50 ('r50') with rbf kernel and a C value of 10 (see sklearn docs):

```python svm.py -f r50 -k rbf -c 10 -g scale```

Train an SVM using a subset of the training data for faster results (e.g. experimenting with regularization value, etc.):

```python svm.py -f r50 -k rbf -c 10 g scale -n 5000   # run with 5000 training examples```

## SVM Hyperparameter Search
Search hyperparameter space for best results. Hyperparameter space is hard-coded in svm.py:

```python svm.py -f r50 -s -n 1000```

## Evaluating Saved Models
Evaluate on validation set:

```python evaluate.py -f r50 -m r50_rbf_10.0_scale_1000000```

Evaluate on test set:

```python evaluate.py -f r50 -m r50_rbf_10.0_scale_1000000 -t```
