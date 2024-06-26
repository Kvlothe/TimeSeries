import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.signal import periodogram
from pmdarima import auto_arima


def check_for_gaps(df):
    # Sort the DataFrame just in case
    df.sort_index(inplace=True)

    # Generate a date range which covers the entire period of your dataset
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

    # Reindex the DataFrame with the date range, filling missing values with NaN
    df_reindexed = df.reindex(date_range)

    # Find gaps where 'Revenue' is NaN within the expected date range
    gaps = df_reindexed[df_reindexed['Revenue'].isnull()]

    print(gaps)


def plot_realization(df):
    # Assuming 'Day' starts from 0 and represents the count of days, we create a date range starting from Jan 1, 2020
    start_date = pd.to_datetime('2020-01-01')  # replace with your actual start date
    df['Date'] = start_date + pd.to_timedelta(df['Day'], unit='D')
    df.set_index('Date', inplace=True)
    df.index.freq = 'D'  # Set the frequency to daily

    # Plotting the revenue over time
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Revenue'], marker='o', linestyle='-')
    plt.title('Revenue over Time')
    plt.xlabel('Day')
    plt.ylabel('Revenue')

    # Improve the x-axis ticks to show months and format dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.grid(True)
    plt.show()


# Function to perform the Dickey-Fuller test
def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def plot_metrics(df):
    # Decompose the time series
    decomposition = seasonal_decompose(df['Revenue'], model='additive')  # or model='multiplicative'

    # Plot the decomposed components
    decomposition.plot()
    plt.show()

    # Plot Autocorrelation
    plot_acf(df['Revenue'])
    plt.show()

    # Plot the ACF and PACF on the differenced data
    plot_acf(df['Revenue_diff'].dropna())
    plt.show()

    plot_pacf(df['Revenue_diff'].dropna())
    plt.show()

    # Spectral Density
    frequencies, spectrum = periodogram(df['Revenue'])
    plt.semilogy(frequencies, spectrum)
    plt.xlabel('Frequency')
    plt.ylabel('Spectral Density')
    plt.show()


# Read in the dataset
df = pd.read_csv("teleco_time_series .csv")

# Get shape of the dataset
# print(df.shape)

# Call function to check for any gaps in the dataset - does each day have a corresponding revenue
# check_for_gaps(df)

# Normalize only the 'Revenue' column
scaler = MinMaxScaler()
df['Revenue'] = scaler.fit_transform(df[['Revenue']])

plot_realization(df)

# Applying first-order differencing
df['Revenue_diff'] = df['Revenue'] - df['Revenue'].shift(1)
test_stationarity(df['Revenue_diff'].dropna())

# Call function to plot the metrics with visuals for Decomposition, Autocorrelation, Spectral Density
plot_metrics(df)

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
df['Revenue'] = df['Revenue'].astype(float)
df.to_csv('cleaned_data.csv')

# Split the data into train and test sets
train = df['Revenue'][:train_size]
test = df['Revenue'][train_size:]

# Perform a stepwise search to find the best SARIMAX model
stepwise_model = auto_arima(train, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=52,
                            start_P=0, start_Q=0, max_P=3, max_Q=3,
                            seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

print(stepwise_model.aic())

model = SARIMAX(df['Revenue'], order=(1,1,0), seasonal_order=(3,1,0,52), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

model_fit = model.fit(disp=False)

# Summarize model results
print(results.summary())

# Check diagnostics
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Forecast the same number of steps as in the test set
forecast = model_fit.get_forecast(steps=len(test))

# Obtain the mean forecast and the confidence intervals
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Calculate the RMSE
rmse = sqrt(mean_squared_error(test, forecast_mean))
print('Test RMSE: %.3f' % rmse)

# Optional: Plot the forecast against the actual values
plt.figure(figsize=(12, 5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast_mean, label='Forecast')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Revenue Forecast vs Actuals')
plt.legend()
plt.show()
