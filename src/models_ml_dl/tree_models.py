import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


def train_test_split(agg_journey_train, agg_journey_test, standardise = False):
    if standardise:
        # standardise the demand per dock
        agg_journey_train['demand_per_dock'] = agg_journey_train['demand'] / agg_journey_train['bike_docks_counts']
        agg_journey_test['demand_per_dock'] = agg_journey_test['demand'] / agg_journey_test['bike_docks_counts']

        # create the target variables
        y_train = agg_journey_train['demand_per_dock']
        y_test = agg_journey_test['demand_per_dock']

        cols_to_remove = ['demand_per_dock']

    else:
        # create the target variables
        y_train = agg_journey_train['demand']
        y_test = agg_journey_test['demand']

        cols_to_remove = ['demand']
    
    # create the predictor variables
    x_train = agg_journey_train.drop(columns=cols_to_remove)
    x_test = agg_journey_test.drop(columns=cols_to_remove)

    return (x_train, y_train, x_test, y_test)



def hyper_param_tuning(model, x_train, y_train, n_iter, n_splits, param_grid):

    tscv = TimeSeriesSplit(n_splits=n_splits)

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=tscv, random_state=42)
    random_search.fit(x_train, y_train)

    return random_search



def add_ma_lag_features(journey_train, journey_test):
    borough_cols = ['start_borough_Hackney', 'start_borough_Islington', 'start_borough_City of London', 'start_borough_Westminster', 'start_borough_Tower Hamlets', 'start_borough_Kensington and Chelsea', 'start_borough_Camden', 'start_borough_Hammersmith and Fulham', 'start_borough_Lambeth', 'start_borough_Wandsworth', 'start_borough_Southwark', 'start_borough_Newham']
    journey_train_ma_lag = pd.DataFrame()
    journey_test_ma_lag = pd.DataFrame()

    for borough in borough_cols:
        # select the data for the current borough
        borough_data_train = journey_train[journey_train[borough] == True].copy()
        borough_data_test = journey_test[journey_test[borough] == True].copy()

        # add moving average features to train data
        borough_data_train['demand_ma_3h'] = borough_data_train['demand'].rolling(3).mean()
        borough_data_train['demand_ma_8h'] =  borough_data_train['demand'].rolling(8).mean()
        borough_data_train['demand_ma_24h'] = borough_data_train['demand'].rolling(24).mean()

        # add lagged demand features to train data
        borough_data_train['demand_lag_1h'] = borough_data_train['demand'].shift(1)
        borough_data_train['demand_lag_8h'] = borough_data_train['demand'].shift(8)
        borough_data_train['demand_lag_24h'] = borough_data_train['demand'].shift(24)
        borough_data_train['demand_lag_1w'] = borough_data_train['demand'].shift(24*7)

        # if there is no data for the first x hours, set actual demand as moving average
        borough_data_train['demand_ma_3h'].fillna(borough_data_train['demand'], inplace=True)
        borough_data_train['demand_ma_8h'].fillna(borough_data_train['demand'], inplace=True)
        borough_data_train['demand_ma_24h'].fillna(borough_data_train['demand'], inplace=True)
        borough_data_train['demand_lag_1h'].fillna(borough_data_train['demand'], inplace=True)
        borough_data_train['demand_lag_8h'].fillna(borough_data_train['demand'], inplace=True)
        borough_data_train['demand_lag_24h'].fillna(borough_data_train['demand'], inplace=True)
        borough_data_train['demand_lag_1w'].fillna(borough_data_train['demand'], inplace=True)

        # append the data for the current borough to the new DataFrame
        journey_train_ma_lag = journey_train_ma_lag.append(borough_data_train)

        # add moving average features to test data
        borough_data_test['demand_ma_3h'] = borough_data_test['demand'].rolling(3).mean()
        borough_data_test['demand_ma_8h'] =  borough_data_test['demand'].rolling(8).mean()
        borough_data_test['demand_ma_24h'] = borough_data_test['demand'].rolling(24).mean()

        # add lagged demand features to test data
        borough_data_test['demand_lag_1h'] = borough_data_test['demand'].shift(1)
        borough_data_test['demand_lag_8h'] = borough_data_test['demand'].shift(8)
        borough_data_test['demand_lag_24h'] = borough_data_test['demand'].shift(24)
        borough_data_test['demand_lag_1w'] = borough_data_test['demand'].shift(24*7)

        # if there is no data for the first x hours, set actual demand as moving average
        borough_data_test['demand_ma_3h'].fillna(borough_data_test['demand'], inplace=True)
        borough_data_test['demand_ma_8h'].fillna(borough_data_test['demand'], inplace=True)
        borough_data_test['demand_ma_24h'].fillna(borough_data_test['demand'], inplace=True)
        borough_data_test['demand_lag_1h'].fillna(borough_data_test['demand'], inplace=True)
        borough_data_test['demand_lag_8h'].fillna(borough_data_test['demand'], inplace=True)
        borough_data_test['demand_lag_24h'].fillna(borough_data_test['demand'], inplace=True)
        borough_data_test['demand_lag_1w'].fillna(borough_data_test['demand'], inplace=True)

        # append the data for the current borough to the new DataFrame
        journey_test_ma_lag = journey_test_ma_lag.append(borough_data_test)

    # sort the train and test data by date and borough
    journey_train_ma_lag.sort_values(['start_date'], inplace=True)
    journey_test_ma_lag.sort_values(['start_date'], inplace=True)

    journey_train_ma_lag.reset_index(drop=True, inplace=True)
    journey_test_ma_lag.reset_index(drop=True, inplace=True)

    return (journey_train_ma_lag, journey_test_ma_lag)



def remove_borough_characteristics(journey_train, journey_test):
    demand = ['demand']
    temporal_features = ['hour', 'part_of_day', 'day_of_week', 'day_of_month', 'day_of_year', 'is_weekend', 'month', 'season', 'bank_holiday']
    weather_features = ['temp', 'feelslike', 'humidity', 'dew', 'precip', 'windgust', 'windspeed', 'cloudcover', 'visibility', 'uvindex']
    borough_features = journey_train.columns[~journey_train.columns.isin(temporal_features) & ~journey_train.columns.isin(weather_features) & ~journey_train.columns.isin(demand)]

    # filter out borough characteristics, keep only temporal and weather features
    journey_train_without_borough_characteristics = journey_train.drop(borough_features, axis=1)
    journey_test_without_borough_characteristics = journey_test.drop(borough_features, axis=1)

    return (journey_train_without_borough_characteristics, journey_test_without_borough_characteristics)




