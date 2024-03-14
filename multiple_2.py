import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Importing Dataset
Data_set = pd.read_csv(r"C:\Users\91805\OneDrive\Desktop\Git\regression\Data2.csv")
x = Data_set.drop(['PE'],axis=1).values
y = Data_set['PE'].values

#spliting the  data into traning and test set.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3,random_state= 0)

#training the Model
ml = LinearRegression()
ml.fit(x_train, y_train)

#predicting the test set result
y_pred = ml.predict(x_test)

print(ml.predict([[5.41,40.07,1019.16,64.77]]))


#Checking R2 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)


