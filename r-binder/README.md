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

## Datasets

The pima-indians-diabetes dataset is partitioned into three separate datasets for this example.

**Training and validation:** pima-indians-diabetes1.csv. 80% of the original dataset is split from the full dataset. In turn, 70% of this dataset is used for training the model, and the remaining 30% is used for validating the predictions.

**Test:** pima-indians-diabetes2.csv and pima-indians-diabetes3.csv. The remaining 20% of the original dataset is used as unseen data, to determine whether the predictions being yielded by the model would perform well when dealing with completely new data. pima-indians-diabetes2 contains the features (or independent variables), while pima-indians-diabetes3 contains the dependent variable (blood glucose levels).

## Feature Selection

The purpose of feature selection is to determine those features that have the most influence on the dependent variable.

In our example, there are eight features - some will be more important than others in determining blood glucose levels.

The two feature selection methods used here are:

- Correlation plots
- Multiple Linear Regression

### Correlation plots

Correlation plots allow us to visually determine:

1. Features that are highly correlated with the dependent variable
2. Features that are highly correlated with each other

If certain features are highly correlated with blood glucose levels, then this is an indication that these features are important in predicting the same. Features with low correlation are indicated to be insignificant.

However, features that are highly correlated with each other would indicate that some of these features are redundant (since they are in effect attempting to explain the same thing).

Here is the first correlation plot:

(correlation plot)

We can see that...

However, we can go into more detail and obtain specific correlation coefficients for each feature:

(correlation plot 2)

X, Y, and Z, are indicated as being relevant features for blood glucose levels.

### Multiple Linear Regression

The purpose of a multiple linear regression is to:

1. Determine the size and nature of the coefficient for each feature in explaining the dependent variable.
2. Determine the signficance or insignificance of each feature.

Here are the results for the linear regression:

(results)

At the 5% level, X, Y, and Z are deemed significant. Other features are deemed insignificant.

Taking the findings of both the correlation plots and multiple linear regression into account, X, Y, and Z are selected as the relevant features for the analysis.

## Data Preparation

Now that the relevant features have been selected, the neural network can be constructed. Before doing so:

1. Max-Min Normalization is used to scale each variable between 0 and 1. This is to ensure a common scale among the variables so that the neural network can interpret them properly.

```
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(df, normalize))
attach(maxmindf)
maxmindf<-as.matrix(maxmindf)
```

2. The train-validation set is split 70/30.

```
ind <- sample(2, nrow(maxmindf), replace=TRUE, prob = c(0.7,0.3))

X_train <- maxmindf[ind==1, 1:4]
X_val <- maxmindf[ind==2, 1:4]
y_train <- maxmindf[ind==1, 5]
y_val <- maxmindf[ind==2, 5]
```
