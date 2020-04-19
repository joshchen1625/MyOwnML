# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:48:57 2019

Implement batch gradient descent

@author: zhuoz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def featureScaling(inputV):
    return (inputV - np.mean(inputV)) / np.std(inputV)

# Gradient Descent
def batchGradientDescent(X, y, alpha = 0.001, threshold = 0.01, loopNumLimit = 5000):
    theta = np.zeros(X.shape[1], dtype = X.dtype)  # Init theta
    m = X.shape[0]  # Number of training examples
    n = X.shape[1]  # Number of features (including the intercept term)
    iterNum = 1
    
    while iterNum <= loopNumLimit:
        print("----------------------", theta)
        s = np.zeros(X.shape[1], dtype = x.dtype)
        for j in range(theta.shape[0]):
            for i in range(m):
#                 print("X[i, :] ", X[i, :])
#                 print("y[i]: ", y[i])
#                 print("theta: ", theta)
#                 print("X[i, j]: ", X[i, j])
#                 print(i, j, (np.dot(X[i, :], theta) - y[i]) * X[i, j])
                s[j] += (np.dot(X[i, :], theta) - y[i]) * X[i, j]
#                 print("~~~~~ s[j]:", s[j])
            s[j] = alpha * s[j]
            print("After update s[j] is: ", s[j])
        if all(abs(element) <= threshold for element in s):
            break
        else:
            print("Before update theta is: ", theta)
            theta = theta - s
            print("After update theta is: ", theta)
            iterNum += 1
    print("Number of iteration is: ", iterNum)
    return theta


# Data set - only take 1 feature from the Portland housing data set
df = pd.read_csv('PortlandHousePrice.txt')
df.head()

LivingArea = df['Living_Area']
Price = df['Price']

x = np.array(LivingArea)
x = featureScaling(x)
x = x.reshape(x.shape[0], 1)
y = np.array(Price)

x_intercept = np.ones((x.shape[0], 1), dtype = x.dtype)
X = np.concatenate((x_intercept, x), axis = 1)  # Design Matrix

# Train the model
updatedtheta = batchGradientDescent(X, y)

# Plot the line
x_Start = LivingArea.min()
x_End = LivingArea.max()
xLine = np.array([x_Start, x_End])

y_Start = updatedtheta[0] + updatedtheta[1] * (xLine[0] - np.mean(LivingArea)) / np.std(LivingArea)
y_End = updatedtheta[0] + updatedtheta[1] * (xLine[1] - np.mean(LivingArea)) / np.std(LivingArea)
yLine = np.array([y_Start, y_End])

plt.scatter(LivingArea, Price)
plt.plot(xLine, yLine)

# Make a prediction
x_Prediction = (x_End - x_Start) / 2 + x_Start
y_Prediction = updatedtheta[0] + updatedtheta[1] * (x_Prediction - np.mean(LivingArea)) / np.std(LivingArea)
print("Prediction, x, y: ", x_Prediction, y_Prediction)