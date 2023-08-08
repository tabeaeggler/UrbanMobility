import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def aggregate_demand(journey_df, aggregation_level):
    """
    Aggregate demand for journeys based on the specified aggregation level.
    
    Parameters:
    journey_df (DataFrame): DataFrame containing the journey data.
    aggregation_level (str): The level at which to aggregate demand, e.g., 'H', 'D', 'W', etc.
    
    Returns:
    DataFrame: The aggregated journey data.
    """

    journey_df = journey_df.copy()

    # aggregate demand for each borough by summing
    journey_df['demand'] = 1

    # create a list of all column names
    column_names = [col for col in journey_df.columns]

    # create a dictionary with all features, first -> always keep values of first element
    aggregate_functions = {col: 'first' for col in column_names}

    # add an entry for the 'demand' column with 'sum' as the aggregate function
    aggregate_functions['demand'] = 'sum'

    # round down the datetime to the specified aggregation level
    journey_df['start_date'] = journey_df['start_date'].dt.floor(aggregation_level)

    # perform the groupby operation
    journey_df_aggregated = journey_df.groupby(['start_date', 'start_borough']).agg(aggregate_functions)


    return journey_df_aggregated.reset_index(drop=True)



def clean_aggregated_df_hourly(agg_jounrey_df, borough_df, temporal_weather_features):
    """
    Clean and preprocess the aggregated journey data for hourly analysis.
    
    Parameters:
    agg_jounrey_df (DataFrame): DataFrame containing the aggregated journey data.
    borough_df (DataFrame): DataFrame containing borough-related features.
    temporal_weather_features (list of str): List of weather features to consider for analysis.
    
    Returns:
    DataFrame: The cleaned and preprocessed DataFrame.
    """

    agg_jounrey_df = agg_jounrey_df.copy()

    # --- 1. add rows for hours where no journeys were started, with demand = 0 --- #
    timestamps = agg_jounrey_df['start_date'].unique()
    boroughs = agg_jounrey_df['start_borough'].unique()

    # create a MultiIndex with all combinations of timestamps and boroughs
    index = pd.MultiIndex.from_product([timestamps, boroughs], names=['start_date', 'start_borough'])
    df = agg_jounrey_df.set_index(['start_date', 'start_borough']).reindex(index).reset_index()

    # fill the demand with 0
    df['demand'] = df['demand'].fillna(0)

    # fill the missing time-specific features with the value from another row with the same timestamp
    df[temporal_weather_features] = df.groupby('start_date')[temporal_weather_features].transform(lambda x: x.fillna(method='ffill'))
    df[temporal_weather_features] = df.groupby('start_date')[temporal_weather_features].transform(lambda x: x.fillna(method='bfill'))

    # fill the borough-related features with the corresponding values from the borough_df for the rows where demand is 0,
    borough_features = agg_jounrey_df.columns[28:102].tolist()

    for feature in borough_features:
        df.loc[df['demand'] == 0, feature] = df['start_borough'].map(borough_df.set_index('borough')[feature])


    # --- 2. create dummies for borough --- #
    df = pd.get_dummies(df, columns=['start_borough'])


    # --- 3. drop unused cols --- #
    cols_to_remove = ['rental_id', 'end_date', 'end_borough', 'end_station_name', 'start_station_name', 'borough', 'borough_code', 'year']
    df = df.drop(columns=cols_to_remove)

    return df


def clean_aggregated_df_daily(agg_jounrey_df):
    """
    Clean and preprocess the aggregated journey data for daily analysis.
    
    Parameters:
    agg_jounrey_df (DataFrame): DataFrame containing the aggregated journey data.
    
    Returns:
    DataFrame: The cleaned and preprocessed DataFrame.
    """

    agg_jounrey_df = agg_jounrey_df.copy()

    # create dummies for borough
    agg_jounrey_df = pd.get_dummies(agg_jounrey_df, columns=['start_borough'])

    # drop unused cols
    cols_to_remove = ['rental_id', 'end_date', 'end_borough', 'end_station_name', 'start_station_name', 'borough', 'borough_code', 'year']
    agg_jounrey_df = agg_jounrey_df.drop(columns=cols_to_remove)

    return agg_jounrey_df



def plot_demand_by_week_borough(df_1, df_2, df_3, df_4, borough):
    """
    Plot demand by week for the specified borough, considering different dataframes for comparison.
    
    Parameters:
    df_1, df_2, df_3, df_4 (DataFrame): DataFrames containing journey data for different years.
    borough (str): The name of the borough for which to plot demand.
    
    Returns:
    DataFrame, DataFrame, DataFrame, DataFrame: Filtered DataFrames for each year specific to the given borough.
    """
        
    # Filter out the data for the specified borough
    borough_data_1 = df_1.loc[df_1[f'start_borough_{borough}'] == 1].copy()
    borough_data_2 = df_2.loc[df_2[f'start_borough_{borough}'] == 1].copy()
    borough_data_3 = df_3.loc[df_3[f'start_borough_{borough}'] == 1].copy()
    borough_data_4 = df_4.loc[df_4[f'start_borough_{borough}'] == 1].copy()

    # Add columns for the date, hour, and week for training data
    borough_data_1.loc[:, 'date'] = borough_data_1['start_date'].dt.date
    borough_data_1.loc[:, 'week_of_year'] = borough_data_1['start_date'].dt.isocalendar().week
    borough_data_1.loc[:, 'hour_of_week'] = borough_data_1['day_of_week'] * 24 + borough_data_1['hour']

    borough_data_2.loc[:, 'date'] = borough_data_2['start_date'].dt.date
    borough_data_2.loc[:, 'week_of_year'] = borough_data_2['start_date'].dt.isocalendar().week
    borough_data_2.loc[:, 'hour_of_week'] = borough_data_2['day_of_week'] * 24 + borough_data_2['hour']

    borough_data_3.loc[:, 'date'] = borough_data_3['start_date'].dt.date
    borough_data_3.loc[:, 'week_of_year'] = borough_data_3['start_date'].dt.isocalendar().week
    borough_data_3.loc[:, 'hour_of_week'] = borough_data_3['day_of_week'] * 24 + borough_data_3['hour']

    borough_data_4.loc[:, 'date'] = borough_data_4['start_date'].dt.date
    borough_data_4.loc[:, 'week_of_year'] = borough_data_4['start_date'].dt.isocalendar().week
    borough_data_4.loc[:, 'hour_of_week'] = borough_data_4['day_of_week'] * 24 + borough_data_4['hour']


    # Aggregate by week and hour of the week for training data
    borough_1_weekly_hourly = borough_data_1.groupby(['week_of_year', 'hour_of_week']).agg({'demand': 'sum'}).reset_index()
    borough_2_weekly_hourly = borough_data_2.groupby(['week_of_year', 'hour_of_week']).agg({'demand': 'sum'}).reset_index()
    borough_3_weekly_hourly = borough_data_3.groupby(['week_of_year', 'hour_of_week']).agg({'demand': 'sum'}).reset_index()
    borough_4_weekly_hourly = borough_data_4.groupby(['week_of_year', 'hour_of_week']).agg({'demand': 'sum'}).reset_index()

    # Calculate the average and standard deviation over all weeks for df1
    mean_weekly_demand_data_1 = borough_1_weekly_hourly.groupby('hour_of_week')['demand'].mean()
    stand_weekly_demand_data_1 = borough_1_weekly_hourly.groupby('hour_of_week')['demand'].std()

    # Create subplots for each week
    fig, axs = plt.subplots(nrows=52, ncols=2, figsize=(30, 90), constrained_layout=True, sharey='row')

    # Plot each week's data for training and test data
    for idx, week in enumerate(borough_1_weekly_hourly['week_of_year'].unique()):
        # Calculate weekly demand
        week_data_1 = borough_1_weekly_hourly[borough_1_weekly_hourly['week_of_year'] == week]  
        week_data_2 = borough_2_weekly_hourly[borough_2_weekly_hourly['week_of_year'] == week]
        week_data_3 = borough_3_weekly_hourly[borough_3_weekly_hourly['week_of_year'] == week]
        week_data_4 = borough_4_weekly_hourly[borough_4_weekly_hourly['week_of_year'] == week]

        # Plot the average and standard deviation from all weeks for training data
        axs[idx, 0].plot(mean_weekly_demand_data_1.index, mean_weekly_demand_data_1, color='orange', label='Average demand 2019 with std. dev.')
        axs[idx, 0].fill_between(stand_weekly_demand_data_1.index, (mean_weekly_demand_data_1 - stand_weekly_demand_data_1), (mean_weekly_demand_data_1 + stand_weekly_demand_data_1),
                            color='orange', alpha=.2)
        
        axs[idx, 0].plot(week_data_1['hour_of_week'], week_data_1['demand'], color='blue', label='2019')

        axs[idx, 0].set_title(f'Week {week} - {borough} (2019)')
        axs[idx, 0].set_ylabel('Demand')
        axs[idx, 0].set_xticks(np.arange(0, 168, 24))  # Set x-axis ticks every 24 hours
        axs[idx, 0].legend(loc="upper right")
        axs[idx, 0].grid(True)

        # Plot test data for the corresponding week
        axs[idx, 1].plot(week_data_4['hour_of_week'], week_data_4['demand'], color='orange', label='2016')     
        axs[idx, 1].plot(week_data_3['hour_of_week'], week_data_3['demand'], color='red', label='2017')       
        axs[idx, 1].plot(week_data_2['hour_of_week'], week_data_2['demand'], color='green', label='2018')
        axs[idx, 1].plot(week_data_1['hour_of_week'], week_data_1['demand'], color='blue', label='2019')

        axs[idx, 1].set_title(f'Week {week} - {borough} (year comparison)')
        axs[idx, 1].set_ylabel('Demand')
        axs[idx, 1].set_xticks(np.arange(0, 168, 24))  # Set x-axis ticks every 24 hours
        axs[idx, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']) 
        axs[idx, 1].legend(loc="upper right")
        axs[idx, 1].grid(True)

    plt.savefig(f'../reports/figures/aggregated_demand_line_graphs/{borough}_demand_by_week.jpg', dpi=300)
    plt.show()

    return borough_data_1, borough_data_2, borough_data_3, borough_data_4
