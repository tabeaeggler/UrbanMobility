import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import matplotlib.dates as mdates



def time_series_decomposition(journey_df, decomposed):

    """
    This function is used for decomposing a time series and visualizing its components.
    
    Parameters:
    journey_df (pandas.DataFrame): The data frame that contains the journey data.
    decomposed (decomposed object): The decomposed time series object.

    Returns:
    matplotlib.pyplot: The plot object that shows the original, original with lockdown periods, trend, seasonality, and residuals of the time series.
    """

    fig, axs = plt.subplots(5, 1, figsize=(14, 12))
    journey_df['date'] = journey_df['start_date'].dt.date
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



def freq_analysis_time_intervals(journey_df):
    """
    Performs a frequency analysis of bike rentals over various time intervals and produces a plot for each interval.
    The time intervals considered are: hour of the day, part of the day, day of the week, month, season, and year.
    
    Parameters:
    journey_df (pandas.DataFrame): DataFrame containing the journey data and: 'hour', 'part_of_day', 'day_of_week', 'month', 'season', and 'year'.

    Returns:
    matplotlib.pyplot: A plot object with subplots for each of the time intervals.
    """
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))

    # hourly bike rentals
    sns.countplot(x='hour', data=journey_df, color='#66c2a5', ax=axs[0, 0])
    axs[0, 0].set_title('Total Bike Rentals by Hour of Day')
    axs[0, 0].set_xticklabels([f'{x:02d}' for x in range(24)])
    axs[0, 0].yaxis.get_major_formatter().set_scientific(False)

    # part of day bike rentals
    sns.countplot(x='part_of_day', data=journey_df, color='#fc8d62', ax=axs[0, 1])
    axs[0, 1].set_title('Total Bike Rentals by Part of Day')
    axs[0, 1].set_xticklabels(['Early morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
    axs[0, 1].yaxis.get_major_formatter().set_scientific(False)

    # daily bike rentals
    sns.countplot(x='day_of_week', data=journey_df, color='#8da0cb', ax=axs[1, 0])
    axs[1, 0].set_title('Total Bike Rentals by Day of Week')
    axs[1, 0].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axs[1, 0].yaxis.get_major_formatter().set_scientific(False)

    # monthly bike rentals
    sns.countplot(x='month', data=journey_df, color='#e78ac3', ax=axs[1, 1])
    axs[1, 1].set_title('Total Bike Rentals by Month')
    axs[1, 1].set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    axs[1, 1].yaxis.get_major_formatter().set_scientific(False)

    # seasonal bike rentals
    sns.countplot(x='season', data=journey_df, color='#a6d854', ax=axs[2, 0])
    axs[2, 0].set_title('Total Bike Rentals by Season')
    axs[2, 0].set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
    axs[2, 0].yaxis.get_major_formatter().set_scientific(False)

    # yearly bike rentals
    sns.countplot(x='year', data=journey_df, color='#ffd92f', ax=axs[2, 1])
    axs[2, 1].set_title('Total Bike Rentals by Year')
    axs[2, 1].yaxis.get_major_formatter().set_scientific(False)

    plt.tight_layout()
    return plt




def demand_analysis_time_intervals(journey_df):
    """
    Performs an analysis of bike rental demand over various time intervals and produces a boxplot for each.
    The time intervals considered are: hour of the day, part of the day, day of the week, month, season, year, weekday, and bank holiday.
    
    Parameters:
    journey_df (pandas.DataFrame): DataFrame containing the journey data,
    containing the columns: 'hour', 'part_of_day', 'day_of_week', 'month', 'season', 'year', 'is_weekend', 'bank_holiday', and 'demand'.
                
    Returns:
    matplotlib.pyplot: A plot object with subplots for each of the time intervals.
    """

    fig, axs = plt.subplots(4, 2, figsize=(18, 12))

    # Hourly bike rentals
    sns.boxplot(x='hour', y='demand', data=journey_df, color='#66c2a5', ax=axs[0, 0])
    axs[0, 0].set_title('Relationship: Demand and Hour of Day')
    axs[0, 0].set_xticklabels([f'{x:02d}' for x in range(24)])
    axs[0, 0].yaxis.get_major_formatter().set_scientific(False)

    # Part of day bike rentals
    sns.boxplot(x='part_of_day', y='demand', data=journey_df, color='#fc8d62', ax=axs[0, 1])
    axs[0, 1].set_title('Relationship: Demand and Part of Day')
    axs[0, 1].set_xticklabels(['Early morning', 'Morning', 'Afternoon', 'Evening', 'Night'])
    axs[0, 1].yaxis.get_major_formatter().set_scientific(False)

    # Daily bike rentals
    sns.boxplot(x='day_of_week', y='demand', data=journey_df, color='#8da0cb', ax=axs[1, 0])
    axs[1, 0].set_title('Relationship: Demand and Day of Week')
    axs[1, 0].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    axs[1, 0].yaxis.get_major_formatter().set_scientific(False)

    # Monthly bike rentals
    sns.boxplot(x='month', y='demand', data=journey_df, color='#e78ac3', ax=axs[1, 1])
    axs[1, 1].set_title('Relationship: Demand and Month')
    axs[1, 1].set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    axs[1, 1].yaxis.get_major_formatter().set_scientific(False)

    # Seasonal bike rentals
    sns.boxplot(x='season', y='demand', data=journey_df, color='#a6d854', ax=axs[2, 0])
    axs[2, 0].set_title('Relationship: Demand and Season')
    axs[2, 0].set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
    axs[2, 0].yaxis.get_major_formatter().set_scientific(False)

    # Yearly bike rentals
    sns.boxplot(x='year', y='demand', data=journey_df, color='#ffd92f', ax=axs[2, 1])
    axs[2, 1].set_title('Relationship: Demand and Year')
    axs[2, 1].set_title('Total Bike Rentals by Year')
    axs[2, 1].yaxis.get_major_formatter().set_scientific(False)

    # Weekday bike rentals
    sns.boxplot(x='is_weekend', y='demand', data=journey_df, color='#e5c494', ax=axs[3, 0])
    axs[3, 0].set_title('Relationship: Demand and Weekday')
    axs[3, 0].set_xticklabels(['Weekday', 'Weekend'])
    axs[3, 0].yaxis.get_major_formatter().set_scientific(False)


    # Bankholiday bike rentals
    sns.boxplot(x='bank_holiday', y='demand', data=journey_df, color='#b3b3b3', ax=axs[3, 1])
    axs[3, 1].set_title('Relationship: Demand and Bank Holiday')
    axs[3, 1].set_xticklabels(['Regular Day', 'Bank Holiday'])
    axs[3, 1].yaxis.get_major_formatter().set_scientific(False)

    plt.tight_layout()
    return plt