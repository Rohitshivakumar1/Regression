import pandas as pd
import matplotlib as plt
import numpy as np

Data_set = pd.read_csv(r"C:\Users\91805\OneDrive\Desktop\Git\regression\Salary_Data.csv")
x = Data_set.iloc[:, :-1].values
y = Data_set.iloc[:, -1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(x_train, y_train)