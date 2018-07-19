

"""house price example linear regression"""
"""import libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""read data"""
data=pd.read_csv("housing.csv")
"""create pandas data frame"""
df=pd.DataFrame(data)
print data
"""initialize dependent and independent variable y dependent and X independent"""
y=df['Price'].values
X=df['SQRFT'].values
plt.scatter(X,y)
plt.xlabel("Square feet")
plt.ylabel("Price of houses")
plt.show()
"""Import sklearn regression model"""
from sklearn.linear_model import LinearRegression
model=LinearRegression(fit_intercept=True)
"""reshaping the data"""
X=df['SQRFT'].values.reshape([-1,1])
"""fit the model"""
model.fit(X,y)
"""model coefficient y=a+Bx1"""
"""a"""
print model.coef_
print model.intercept_

x_fit=np.linspace(256,260)
print x_fit
X_fit=x_fit.reshape(-1,1)
"""Prediction"""
y_fit=model.predict(X_fit)
"""prediction from 256 to 260 square feet"""
print (y_fit)
"""prediction price of 100 square feet house"""

print(model.predict(100))
"""scatter plot"""
plt.scatter(X,y)
plt.plot(x_fit,y_fit)
plt.show()

from sklearn.externals import joblib
joblib.dump(model, 'model.pkl')