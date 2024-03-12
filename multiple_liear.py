import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Data_set = pd.read_csv(r"C:\Users\91805\OneDrive\Desktop\Git\regression\50_Startups.csv")
x = Data_set.iloc[:, :-1]
y = Data_set.iloc[:, -1]
print(x)




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x_transformed = ct.fit_transform(x)
x_transformed[:, 3] = x_transformed[:, 3].round(1)
x = pd.DataFrame(x_transformed)

print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.values.reshape(len(y_test), 1)),1))
