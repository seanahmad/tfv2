# Section 4 - TensorFlow 2.0 essentials: What's new (R)

With the advent of TensorFlow 2.0, *Keras* is now the default API for this version. Keras is used to build neural networks for deep learning purposes. As such, Keras is a highly useful tool for conducting analysis on large datasets.

However, did you realise that the Keras API can also be run in R?

In this example, *Keras* is used to generate a neural network - with the aim of solving a regression problem in R.

Specifically, the [Pima Indians Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) is used in order to predict blood glucose levels for patients using the relevant features.

In this regard, this article provides an overview of:

- Feature selection methods in R
- How to define a Sequential model in Keras
- Methods to validate and test model predictions

## Datasets

The Pima Indians Diabetes dataset is partitioned into three separate datasets for this example.

**Training and validation:** *pima-indians-diabetes1.csv*. 80% of the original dataset is split from the full dataset. In turn, 70% of this dataset is used for training the model, and the remaining 30% is used for validating the predictions.

**Test:** *pima-indians-diabetes2.csv* and *pima-indians-diabetes3.csv*. The remaining 20% of the original dataset is used as unseen data, to determine whether the predictions being yielded by the model would perform well when dealing with completely new data. *pima-indians-diabetes2* contains the features (or independent variables), while *pima-indians-diabetes3* contains the dependent variable (blood glucose levels).

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

```
M <- cor(diabetes1)
corrplot(M, method = "circle")
```

![corrplot](corrplot.png)

We can see that the **Insulin** and **Outcome** variables are particularly correlated with the **Glucose** variable, while there is also correlation between **Age** and **Pregnancies** and **Insulin** and **Skin Thickness**.

However, we can go into more detail and obtain specific correlation coefficients for each feature:

```
corrplot(M, method = "number")
```

![corrplot-with-stats](corrplot-with-stats.png)

### Multiple Linear Regression

The purpose of a multiple linear regression is to:

1. Determine the size and nature of the coefficient for each feature in explaining the dependent variable.
2. Determine the signficance or insignificance of each feature.

Here are the results for the linear regression:

```
Call:
lm(formula = Glucose ~ Pregnancies + Outcome + Age + DiabetesPedigreeFunction + 
    BMI + Insulin + SkinThickness + BloodPressure, data = diabetes1)

Residuals:
    Min      1Q  Median      3Q     Max 
-68.709 -18.148  -2.212  15.176  80.950 

Coefficients:
                          Estimate Std. Error t value Pr(>|t|)    
(Intercept)              78.401064   6.363612  12.320  < 2e-16 ***
Pregnancies              -0.481865   0.363730  -1.325  0.18575    
Outcome                  25.590805   2.384153  10.734  < 2e-16 ***
Age                       0.527262   0.106097   4.970  8.8e-07 ***
DiabetesPedigreeFunction  0.052534   3.198192   0.016  0.98690    
BMI                       0.318452   0.167106   1.906  0.05718 .  
Insulin                   0.082208   0.009843   8.352  4.8e-16 ***
SkinThickness            -0.202236   0.077372  -2.614  0.00918 ** 
BloodPressure             0.083865   0.058081   1.444  0.14929    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 24.94 on 590 degrees of freedom
Multiple R-squared:  0.362,	Adjusted R-squared:  0.3533 
F-statistic: 41.84 on 8 and 590 DF,  p-value: < 2.2e-16
```

At the 5% level, **Outcome**, **Age**, **Insulin** and **Skin Thickness** are deemed significant. Other features are deemed insignificant at the 5% level.

## Heteroscedasticity test using Breusch-Pagan

It is not deemed necessary to run a formal test for multicollinearity in this instance, as the correlation plots indicate features that are highly correlated with each other.

However, heteroscedasticity (uneven variance across standard errors) could be present, e.g. due to differing age across patients. In order to test this, the Breusch-Pagan test is run - with a p-value below 0.05 indicating the presence of heteroscedasticity.

```
> bptest(fit)

	studentized Breusch-Pagan test

data:  fit
BP = 36.585, df = 8, p-value = 1.372e-05
```

As heteroscedasticity is indicated to be present, a robust regression is run - specifically using Huber weights. The purpose of this is to place less value on the outliers present in the dataset.

```
> # Huber Weights (Robust Regression)
> summary(rr.huber <- rlm(Glucose ~ Pregnancies + Outcome + Age + DiabetesPedigreeFunction + BMI + Insulin + SkinThickness + BloodPressure, data=diabetes1))

Call: rlm(formula = Glucose ~ Pregnancies + Outcome + Age + DiabetesPedigreeFunction + 
    BMI + Insulin + SkinThickness + BloodPressure, data = diabetes1)
Residuals:
    Min      1Q  Median      3Q     Max 
-68.627 -16.842  -1.543  15.576  83.793 

Coefficients:
                         Value   Std. Error t value
(Intercept)              78.3319  6.2990    12.4357
Pregnancies              -0.4675  0.3600    -1.2984
Outcome                  25.0513  2.3599    10.6152
Age                       0.5448  0.1050     5.1881
DiabetesPedigreeFunction -0.5482  3.1657    -0.1732
BMI                       0.3297  0.1654     1.9935
Insulin                   0.0925  0.0097     9.4912
SkinThickness            -0.2530  0.0766    -3.3032
BloodPressure             0.0673  0.0575     1.1706

Residual standard error: 24.53 on 590 degrees of freedom
```

On **590** degrees of freedom, the two-tailed t critical value is as follows:

```
> abs(qt(0.05/2, 590))
[1] 1.963993
```

When the t statistic > t critical value, the null hypothesis is rejected. In this regard, **Outcome**, **Age**, **BMI**, **Insulin**, and **Skin Thickness** have an absolute t-value greater than the critical value.

Taking the findings of both the correlation plots and multiple linear regression into account, **Outcome**, **Age**, **Insulin** and **Skin Thickness** are selected as the relevant features for the analysis.

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

Now, the Sequential model is defined. The four input features (Outcome, Age, Insulin, Skin Thickness) are included in the input layer with 9 neurons defined in the layer. One hidden layer with 60 neurons is defined, and a linear output layer with 1 neuron is defined.

As explained in this article by [Farhad Malik](https://medium.com/fintechexplained/what-are-hidden-layers-4f54f7328263), the number of neurons in each layer is configured as follows:

- **Input layer:** The number of neurons in the input layer is calculated as follows:

```Number of features in the training set + 1```

In this case, as there were 8 features in the training set to begin with, **9** input neurons are defined accordingly.

- **Hidden layer:** One hidden layer is defined, as a single layer is suitable when working with most datasets. The number of neurons in the hidden layer is determined as follows:

```
Training Data Samples/Factor * (Input Neurons + Output Neurons)
```

A factor of 1 is set in this case, the purpose of the factor being to prevent overfitting. A factor can take a value between 1 and 10. With 9 neurons in the input layer, 1 neuron in the output layer and 599 observations in the training set, the hidden layer is assigned 60 neurons.

- **Output layer:** As this is the result layer, the output layer takes a value of 1 by default.

```
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 9, activation = 'relu', kernel_initializer='RandomNormal', input_shape = c(4)) %>% 
  layer_dense(units = 60, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'linear')

summary(model)
```

Here is the output:

```
Model: "sequential"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
dense (Dense)                       (None, 9)                       45          
________________________________________________________________________________
dense_1 (Dense)                     (None, 60)                      600         
________________________________________________________________________________
dense_2 (Dense)                     (None, 1)                       61          
================================================================================
Total params: 706
Trainable params: 706
Non-trainable params: 0
________________________________________________________________________________
```

The model is now trained over 30 epochs, and evaluated based on its loss and mean absolute error. Given that the dependent variable is interval, the mean squared error is used to determine the deviation between the predictions and actual values.

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
    0.0266957393988254
$mae
    0.132186755537987

Model
Model: "sequential"
________________________________________________________________________________
Layer (type)                        Output Shape                    Param #     
================================================================================
dense (Dense)                       (None, 9)                       45          
________________________________________________________________________________
dense_1 (Dense)                     (None, 60)                      600         
________________________________________________________________________________
dense_2 (Dense)                     (None, 1)                       61          
================================================================================
Total params: 706
Trainable params: 706
Non-trainable params: 0
________________________________________________________________________________
```

Here is a plot of the loss and mean absolute error:

![loss](loss.png)

The model yields a loss of just above 2% and a mean absolute error of just above 13%.

The *MLmetrics* library can also be used to calculate the MAPE (mean absolute percentage error).

```
install.packages("MLmetrics")
library(MLmetrics)
MAPE(predicted, actual)
```

The MAPE for the validation set comes in at **18%**. Increasing the number of hidden layers in the model did not improve MAPE and it was therefore decided to keep one hidden layer in the model configuration.

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

![predicted-vs-actual](predicted-vs-actual.png)

Now, the mean absolute percentage error is calculated using the test values:

```
MAPE(predicted_test, actual_test)
```

A mean percentage error of 17% is calculated:

```
0.177895157636775
```

It is observed that while the mean percentage error is slightly higher than that calculated using the training and validation data, the model still performs well in predicting blood glucose levels across unseen observations on the test set.

## Conclusion

In this example, we have seen:

- How to implement feature selection methods in R
- Construct a neural network to analyse regression data using the Keras API
- Gauge prediction accuracy using test data

Many thanks for your time!
