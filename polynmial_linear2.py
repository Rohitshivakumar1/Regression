import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

x = np.random.rand(100,1)
y = 4 + 2 * x + 5 * x**2 +  np.random.randn(100,1)


poly = PolynomialFeatures(degree=3,include_bias=False)
x_poly = poly.fit_transform(x)

reg = LinearRegression()
reg.fit(x_poly,y)

x_vals = np.linspace(-2, 2, 100).reshape(-1,1)
x_vals_poly = poly.transform(x_vals)
y_vals = reg.predict(x_vals_poly)

plt.scatter(x,y)
plt.plot(x_vals,y_vals, color = 'red')
plt.show()