import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#importing Dataset
data_set = pd.read_csv(r"C:\Users\91805\OneDrive\Desktop\Git\regression\Position_Salaries.csv")
x = data_set.iloc[:, 1:-1].values
y = data_set.iloc[:, -1].values

#Traning the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Traning polynomial regression model on whole dataset
poly = PolynomialFeatures(degree= 4)
poly_x = poly.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_x,y)

#Visualing the linear regression results
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title("truth or bluff")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()


#Visualing the linear regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(poly.fit_transform(x)),color = 'blue')
plt.title("truth or bluff(polynomial Regression)")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()


print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly.fit_transform([[6.5]])))