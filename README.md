
# EX.NO.09    A project on Time series analysis on weather forecasting using ARIMA model 
###  DATE:
### AIM:
To Create a project on Time series analysis on coffeesales forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of furniture sales.
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the data with the appropriate encoding
data = pd.read_csv("/content/Super_Store_data.csv", encoding='ISO-8859-1')

# Convert the 'Order Date' column to datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Extract date from 'Order Date' and group by date, summing the 'Sales' column
data['date'] = data['Order Date'].dt.date
daily_sales = data.groupby('date')['Sales'].sum().reset_index()

# Set date as the index
daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales.set_index('date', inplace=True)

def arima_model(data, target_variable, order):
    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Define and fit the ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Generate forecasts for the test period
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE to assess forecast accuracy
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot the training data, testing data, and forecast
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Call the ARIMA model on the daily sales data
arima_model(daily_sales, 'Sales', order=(5, 1, 0))

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/663edec0-bf29-4819-a2e0-3c9af0a94985)



### RESULT:
Thus the program run successfully based on the ARIMA model using python.
