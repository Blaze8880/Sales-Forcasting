import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load sales data (assuming daily data)
sales_data = pd.read_csv('sales_data.csv', parse_dates=['Date'], index_col='Date')

# Resample to monthly sales
monthly_sales = sales_data['Sales'].resample('M').sum()

# Train-test split
train = monthly_sales[:-12]
test = monthly_sales[-12:]

# Build ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast for the test period
forecast = model_fit.forecast(steps=12)
mae = mean_absolute_error(test, forecast)

print(f'Mean Absolute Error: {mae}')

# Plot actual vs. forecasted values (optional)
