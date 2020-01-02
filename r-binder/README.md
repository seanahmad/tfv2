# Section 4 - TensorFlow 2.0 essentials: What's new (R)

With the advent of TensorFlow 2.0, *Keras* is now the default API for this version. Keras is used to build neural networks for deep learning purposes. As such, Keras is a highly useful tool for conducting analysis on large datasets.

However, did you realise that the Keras API can also be run in R?

In this example, *Keras* is used to generate a neural network - with the aim of solving a regression problem in R.

Specifically, the pima-indians-diabetes dataset is used in order to predict blood glucose levels for patients using the relevant features.

In this regard, this article provides an overview of:

- Feature selection methods in R
- How to define a Sequential model in Keras
- Methods to validate and test model predictions
- Advantages and disadvantages of running Keras using the R interface

## Data Preparation

The pima-indians-diabetes dataset is partitioned into three separate datasets for this example.

**Training and validation:** pima-indians-diabetes1.csv. 80% of the original dataset is split from the full dataset. In turn, 70% of this dataset is used for training the model, and the remaining 30% is used for validating the predictions.

**Test:** pima-indians-diabetes2.csv and pima-indians-diabetes3.csv. The remaining 20% of the original dataset is used as unseen data, to determine whether the predictions being yielded by the model would perform well when dealing with completely new data. pima-indians-diabetes2 contains the features (or independent variables), while pima-indians-diabetes3 contains the dependent variable (blood glucose levels).

