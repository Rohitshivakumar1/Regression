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
x_transformed_df = pd.DataFrame(x_transformed)
x = x_transformed_df

print(x)
