import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import linear_model


ds = pd.read_csv(r'C:\Users\91805\OneDrive\Desktop\Git\regression\homeprices.csv')
x = ds.iloc[:, :-1]
y = ds.iloc[:, -1]



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
x[['bedrooms']] = imputer.fit_transform(x[['bedrooms']])
meadian_bedrooms = np.floor(x['bedrooms'].median())
x['bedrooms'].fillna(meadian_bedrooms, inplace= True)
print(x)

reg = linear_model.LinearRegression()
reg.fit(x[['area', 'bedrooms', 'age']], y)
print('coefficients', reg.coef_)
square_feet = int(input("Enter square feet of property "))
bedrooms = int(input("number of bedrooms?"))
age = int(input("age of property?"))


    
new_data = np.array([[square_feet, bedrooms, age]])
estimated_price = reg.predict(new_data)
rounded_price = round(estimated_price[0], 2)
print("Estimated price:", rounded_price)