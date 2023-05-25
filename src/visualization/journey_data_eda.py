import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
import calmap
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


def freq_analysis_time_intervals(journey_df):
    fig, axs = plt.subplots(3, 2, figsize=(19, 16))

    # Hourly bike rentals
    sns.countplot(x='hour', data=journey_df, color='#66c2a5', ax=axs[0, 0])
    axs[0, 0].set_title('Total Bike Rentals by Hour of Day')
    axs[0, 0].set_xticklabels([f'{x:02d}' for x in range(24)])
    axs[0, 0].yaxis.get_major_formatter().set_scientific(False)

    # Part of day bike rentals
    sns.countplot(x='part_of_day', data=journey_df, color='#fc8d62', ax=axs[0, 1])
    axs[0, 1].set_title('Total Bike Rentals by Part of Day')
    axs[0, 1].set_xticklabels(['Early morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
    axs[0, 1].yaxis.get_major_formatter().set_scientific(False)

    # Daily bike rentals
    sns.countplot(x='day_of_week', data=journey_df, color='#8da0cb', ax=axs[1, 0])
    axs[1, 0].set_title('Total Bike Rentals by Day of Week')
    axs[1, 0].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axs[1, 0].yaxis.get_major_formatter().set_scientific(False)

    # Monthly bike rentals
    sns.countplot(x='month', data=journey_df, color='#e78ac3', ax=axs[1, 1])
    axs[1, 1].set_title('Total Bike Rentals by Month')
    axs[1, 1].set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    axs[1, 1].yaxis.get_major_formatter().set_scientific(False)

    # Seasonal bike rentals
    sns.countplot(x='season', data=journey_df, color='#a6d854', ax=axs[2, 0])
    axs[2, 0].set_title('Total Bike Rentals by Season')
    axs[2, 0].set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
    axs[2, 0].yaxis.get_major_formatter().set_scientific(False)

    # Yearly bike rentals
    sns.countplot(x='year', data=journey_df, color='#ffd92f', ax=axs[2, 1])
    axs[2, 1].set_title('Total Bike Rentals by Year')
    axs[2, 1].yaxis.get_major_formatter().set_scientific(False)


    plt.tight_layout()
    return plt


def time_series_decomposition(journey_df):

    # Set the size of the figure
    fig, axs = plt.subplots(5, 1, figsize=(14, 12))

    # Create a 'date' column that contains only the date part of the 'start_time'
    journey_df['date'] = journey_df['start_date'].dt.date

    # Group by date and count the number of journeys
    journeys_per_date = journey_df.groupby(['date']).size().reset_index(name='count')

    # 1. Plot the original data
    axs[0].plot(decomposed.observed, label='Original')
    axs[0].legend(loc='upper left')
    axs[0].set_title("Observed")

    # 2. Plot the original data with lockdown periods
    axs[1].plot(journeys_per_date['date'], journeys_per_date['count'], label='Original with lockdown periods')
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].axvspan('2020-03-23', '2020-07-03', color='orange', alpha=0.3, label='First National Lockdown')
    axs[1].axvspan('2020-11-05', '2020-11-26', color='red', alpha=0.3, label='Second National Lockdown')
    axs[1].axvspan('2021-01-06', '2021-03-08', color='purple', alpha=0.3, label='Third National Lockdown')
    axs[1].legend()
    axs[1].set_title('Observed with lockdown periods')


    # 3. Plot the trend
    axs[2].plot(decomposed.trend, label='Trend')
    axs[2].legend(loc='upper left')
    axs[2].set_title("Trend")

    # 4. Plot the seasonality
    axs[3].plot(decomposed.seasonal, label='Seasonality')
    axs[3].legend(loc='upper left')
    axs[3].set_title("Seasonality")

    # 5. Plot the residuals
    axs[4].plot(decomposed.resid, label='Residuals')
    axs[4].legend(loc='upper left')
    axs[4].set_title("Residuals")

    plt.tight_layout()
    return plt
