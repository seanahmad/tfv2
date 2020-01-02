# Section 4 - TensorFlow 2.0 essentials: What's new (R)

With the advent of TensorFlow 2.0, *Keras* is now the default API for this version. Keras is used to build neural networks for deep learning purposes. As such, Keras is a highly useful tool for conducting analysis on large datasets.

However, did you realise that the Keras API can also be run in R?

In this example, *Keras* is used to generate a neural network - with the aim of solving a regression problem in R.

Specifically, the pima-indians-diabetes dataset is used in order to predict blood glucose levels for patients using the relevant features.

In this regard, this article provides an overview of:

- Feature selection methods in R
- How to define a Sequential model in Keras
- Methods to validate and test model predictions

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

![corrplot](corrplot.png)

We can see that...

However, we can go into more detail and obtain specific correlation coefficients for each feature:

![corrplot-with-stats](corrplot-with-stats.png)

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

## Sequential Model

Now, the Sequential model is defined. The four input features (Outcome, Age, Insulin, Skin Thickness) are included in the input layer. One hidden layer is defined, and a linear output layer is defined.

```
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 12, activation = 'relu', kernel_initializer='RandomNormal', input_shape = c(4)) %>% 
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'linear')

summary(model)
```

Here is the output:

```
Model: "sequential"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
dense (Dense)                       (None, 12)                      60          
________________________________________________________________________________
dense_1 (Dense)                     (None, 8)                       104         
________________________________________________________________________________
dense_2 (Dense)                     (None, 1)                       9           
================================================================================
Total params: 173
Trainable params: 173
Non-trainable params: 0
________________________________________________________________________________
```

The model is now trained over 60 epochs, and evaluated based on its loss and mean absolute error. Given that the dependent variable is interval, the mean squared error is used to determine the deviation between the predictions and actual values.

```
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('mae')
)

history <- model %>% fit(
  X_train, y_train, 
  epochs = 30, batch_size = 50, 
  validation_split = 0.2
)
```

### Model Evaluation

The predicted and actual values are scaled back to their original formats:

```
model %>% evaluate(X_val, y_val)
model
pred <- data.frame(y = predict(model, as.matrix(X_val)))
predicted=pred$y * abs(diff(range(df$Glucose))) + min(df$Glucose)
actual=y_val * abs(diff(range(df$Glucose))) + min(df$Glucose)
df<-data.frame(predicted,actual)
attach(df)
```

Here is the output:

```
$loss
0.0281546051063907
$mae
0.139013439416885
Model
Model: "sequential"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
dense (Dense)                       (None, 12)                      60          
________________________________________________________________________________
dense_1 (Dense)                     (None, 8)                       104         
________________________________________________________________________________
dense_2 (Dense)                     (None, 1)                       9           
================================================================================
Total params: 173
Trainable params: 173
Non-trainable params: 0
________________________________________________________________________________
```

Here is a plot of the loss and mean absolute error:

![loss](loss.png)

The model yields a loss of just under 3% and a mean absolute error of just under 14%.

The mean percentage error is also calculated:

```
mpe=((predicted-actual)/actual)
mean(mpe)*100
```

The MPE is calculated as being just under 4%:

```
3.6604357976026
```

## Predictions and Test Data

Even though the model has shown strong predictive power, our work is not done yet.

While the model has performed well on the validation data, we now need to assess whether the model will also perform well on completely unseen data.

The feature variables are loaded from pima-indians-diabetes2, and max0min normalization is invoked once again:

```
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf2 <- as.data.frame(lapply(df2, normalize))
attach(maxmindf2)
```

Using the predict function in R, predictions are generated for the Glucose variable:

```
pred_test <- data.frame(y = predict(model, as.matrix(maxmindf2)))
predicted_test = pred_test$y * abs(diff(range(diabetes1$Glucose))) + min(diabetes1$Glucose)
predicted_test
```

The predicted values are then compared to the actual values in pima-indians-diabetes3:

```
actual_test = diabetes3$Glucose
df2<-data.frame(predicted_test,actual_test)
attach(df2)
df2
```

(predicted-vs-actual-test)

Now, the mean percentage error is calculated using the test values:

```
mpe2=((predicted_test-actual_test)/actual_test)
mean(mpe2)*100
```

A mean percentage error of just over 5% is calculated:

```
5.31887562145206
```

## Conclusion

