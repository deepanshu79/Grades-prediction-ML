# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset1 = pd.read_csv('Linear_X_Train.csv')
X_train = dataset1.iloc[:, :].values
dataset2 = pd.read_csv('Linear_Y_Train.csv')
Y_train = dataset2.iloc[:, :].values
dataset3 = pd.read_csv('Linear_X_Test.csv')
X_test = dataset3.iloc[:, :].values

# Fitting Linear Regression model to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# predicting the test set results
y_pred = regressor.predict(X_test)

# visualing the training set results
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Time spent on coding VS Grades')
plt.xlabel('Time spent on coding')
plt.ylabel('Grades')
plt.show()

