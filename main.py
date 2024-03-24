import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller


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


def check_stationarity(df):
    adf_result = adfuller(df['Revenue'])

    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))

    # Interpretation
    if adf_result[0] < adf_result[4]["5%"]:
        print("The time series is stationary at a 5% level.")
    else:
        print("The time series is not stationary at a 5% level.")


def plot_realization(df):
    # Assuming 'Day' starts from 0 and represents the count of days, we create a date range starting from Jan 1, 2020
    start_date = pd.to_datetime('2020-01-01')
    df['Day'] = pd.date_range(start=start_date, periods=len(df), freq='D')
    df.set_index('Day', inplace=True)

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


# Read in the dataset
df = pd.read_csv("teleco_time_series .csv")

# Get shape of the dataset
# print(df.shape)

check_for_gaps(df)

# Normalize only the 'Revenue' column
scaler = MinMaxScaler()
df['Revenue'] = scaler.fit_transform(df[['Revenue']])

plot_realization(df)

check_stationarity(df)
