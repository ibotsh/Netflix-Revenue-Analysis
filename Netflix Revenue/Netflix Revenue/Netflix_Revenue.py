# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
netRev = pd.read_csv('netflix_revenue.csv')

# Exploring column names
print(netRev.columns)

# Exploring the dataset
print(netRev.head())

# Convert the 'Date' column to datetime format
netRev['Date'] = pd.to_datetime(netRev['Date'], format='%d-%m-%Y')

# Adding numerical value of the date for modeling 
netRev['Date_ordinal'] = netRev['Date'].map(pd.Timestamp.toordinal)

# Plotting data to find trends
plt.scatter(netRev['Date'], netRev['UCAN Streaming Revenue'], color='blue')
plt.title('US and Canada Streaming Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('US and Canada Streaming Revenue')
plt.xticks(rotation=45)
plt.show()

# Linear Regression on unscaled data
x = netRev['Date_ordinal'].values.reshape(-1, 1)
y = netRev['UCAN Streaming Revenue'].values.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y)

print("Coefficient:", regressor.coef_)
print("Intercept:", regressor.intercept_)

# Plot the regression line on unscaled data
plt.scatter(netRev['Date'], netRev['UCAN Streaming Revenue'], color='blue', label='Actual Data')
plt.plot(netRev['Date'], regressor.predict(x), color='red', label='Regression Line')
plt.title('US and Canada Streaming Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('US and Canada Streaming Revenue')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Feature scaling to improve model performance
scaler = StandardScaler()
scaled_dates = scaler.fit_transform(netRev[['Date_ordinal']])
scaled_revenue = scaler.fit_transform(netRev[['UCAN Streaming Revenue']])

# Fit the linear regression model on scaled data
regressor.fit(scaled_dates, scaled_revenue)

# Plot the regression line with scaled data
plt.scatter(scaled_dates, scaled_revenue, color='blue', label='Scaled Data')
plt.plot(scaled_dates, regressor.predict(scaled_dates), color='red', label='Regression Line')
plt.title('Scaled: US and Canada Streaming Revenue Over Time')
plt.xlabel('Scaled Dates')
plt.ylabel('Scaled Revenue')
plt.legend()
plt.show()

