setwd("C:/Users/michaeljgrogan/Documents/Desktop")
diabetes1<-read.csv("pima-indians-diabetes1.csv")
attach(diabetes1)
library(lmtest)
library("MASS")

# OLS Regression Results

fit <- lm(Glucose ~ Pregnancies + Outcome + Age + DiabetesPedigreeFunction + BMI + Insulin + SkinThickness + BloodPressure, data=diabetes1)
summary(fit) # show results

# Breusch-Pagan Test for Heteroscedasticity
bptest(fit)

# Huber Weights (Robust Regression)
summary(rr.huber <- rlm(Glucose ~ Pregnancies + Outcome + Age + DiabetesPedigreeFunction + BMI + Insulin + SkinThickness + BloodPressure, data=diabetes1))
abs(qt(0.05/2, 590))