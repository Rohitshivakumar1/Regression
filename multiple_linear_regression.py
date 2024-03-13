import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
ds = pd.read_csv(r'C:\Users\91805\OneDrive\Desktop\Git\regression\homeprices.csv')

# Split dataset into features (X) and target variable (y)
X = ds[['area', 'bedrooms', 'age']]
y = ds['price']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
square_feet = int(input("Enter square feet of property: "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter age of property: "))

new_data = np.array([[square_feet, bedrooms, age]])
new_data_scaled = scaler.transform(new_data)
estimated_price = model.predict(new_data_scaled)
print("Estimated price:", estimated_price[0])

# Evaluate model performance
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)