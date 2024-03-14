import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

#Importing the Dataset
df = pd.read_csv(r"C:\Users\91805\OneDrive\Desktop\Git\regression\homeprices.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Encoding the categerical model
lb = LabelEncoder()
x[:, 2] = lb.fit_transform(x[:, 2])
one = OneHotEncoder()
x = one.fit_transform(x).toarray()

#Avoiding dummy variable trap
x = x[:, 1:]

#spliting data for test and traning sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)


# Fitting multiple linear regression to training set
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set result
y_pred = regressor.predict(x_test)

# Building the model using backward elimination
num_rows = x_train.shape[0]
ones_column = np.ones((num_rows, 1)).astype(int)
x_train = np.append(arr=ones_column, values=x_train, axis=1)

num_rows_test = x_test.shape[0]
ones_column_test = np.ones((num_rows_test, 1)).astype(int)
x_test = np.append(arr=ones_column_test, values=x_test, axis=1)

print(x_train)
print(x_test)