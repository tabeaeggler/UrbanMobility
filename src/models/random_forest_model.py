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

        cols_to_remove = ['rental_id', 'end_date', 'end_borough', 'start_date', 'end_station_name', 'start_station_name', 'demand', 'borough', 'borough_code', 'year', 'start_borough', 'start_date_hour', 'demand_per_dock']

    else:
        # create the target variables
        y_train = agg_journey_train['demand']
        y_test = agg_journey_test['demand']

        cols_to_remove = ['rental_id', 'end_date', 'end_borough', 'start_date', 'end_station_name', 'start_station_name', 'demand', 'borough', 'borough_code', 'year', 'start_borough', 'start_date_hour']
    
    # create the predictor variables
    x_train = agg_journey_train.drop(columns=cols_to_remove)
    x_test = agg_journey_test.drop(columns=cols_to_remove)

    return (x_train, y_train, x_test, y_test)





def random_forest_fit_pred(x_train, y_train, x_test):

    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    return (rf, y_pred)



def gradient_boosting_fit_pred(x_train, y_train, x_test):
    from sklearn.ensemble import GradientBoostingRegressor

    gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, random_state=42)

    gb.fit(x_train, y_train)
    y_pred = gb.predict(x_test)

    return (gb, y_pred)



def get_feature_importance(rf, x_train):

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)

    return feature_importances



def evaluation_metrics(y_test, y_pred):

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return (rmse, mae, r2)



def evaluation_vis(y_test, y_pred):

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Actual vs Predicted
    axs[0].scatter(y_test, y_pred, alpha=0.3)
    axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axs[0].set_xlabel('Actual')
    axs[0].set_ylabel('Predicted')
    axs[0].set_title('Actual vs Predicted')

    # Plot 2: Residuals vs Predicted
    residuals = y_test - y_pred
    axs[1].scatter(y_pred, residuals, alpha=0.3)
    axs[1].axhline(0, color='k', linestyle='--', lw=2)
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('Residuals')
    axs[1].set_title('Residuals vs Predicted')

    # Plot 3: Histogram of Residuals
    axs[2].hist(residuals, bins=100)
    axs[2].axvline(0, color='k', linestyle='--', lw=2)
    axs[2].set_xlabel('Residuals')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('Histogram of Residuals')

    plt.tight_layout()
    plt.show()


def evaluation_actual_vs_predicted(y_test, y_pred, x_test, borough, folder):
    # Create a copy of the test DataFrame and add predicted demand
    evaluation_df = x_test.copy()
    evaluation_df['actual_demand'] = y_test
    evaluation_df['predicted_demand'] = y_pred

    # Filter the data for the given borough
    if borough != 'All_Boroughs':
        evaluation_df = evaluation_df[evaluation_df[f'start_borough_{borough}'] == 1]

    # Create date, day of week, week of year and hour of week columns
    evaluation_df['start_date_hour'] = pd.to_datetime(evaluation_df['start_date_hour'])
    evaluation_df['date'] = evaluation_df['start_date_hour'].dt.date
    evaluation_df['day_of_week'] = evaluation_df['start_date_hour'].dt.dayofweek
    evaluation_df['week_of_year'] = evaluation_df['start_date_hour'].dt.isocalendar().week
    evaluation_df['hour_of_week'] = evaluation_df['day_of_week'] * 24 + evaluation_df['start_date_hour'].dt.hour
    
    # Aggregate by week and hour of the week
    evaluation_weekly_hourly = evaluation_df.groupby(['week_of_year', 'hour_of_week']).agg({'actual_demand': 'sum', 'predicted_demand': 'sum'}).reset_index()

    # Create subplots for each week
    fig, axs = plt.subplots(nrows=52, figsize=(30, 70), constrained_layout=True, sharey='row')

    # Plot each week's data for actual and predicted demand
    for idx, week in enumerate(evaluation_weekly_hourly['week_of_year'].unique()):
        # Extract data for this week
        week_data = evaluation_weekly_hourly[evaluation_weekly_hourly['week_of_year'] == week]
        
        # Plot demand for this week
        axs[idx].plot(week_data['hour_of_week'], week_data['predicted_demand'], color='red', label='Predicted demand')
        axs[idx].plot(week_data['hour_of_week'], week_data['actual_demand'], color='blue', label='Actual demand (test, 2019)')

        axs[idx].set_title(f'Week {week} - {borough} (Actual vs. Predicted)')
        axs[idx].set_ylabel('Demand')
        axs[idx].set_xticks(np.arange(0, 168, 24))  # Set x-axis ticks every 24 hours
        axs[idx].legend(loc="upper right")
        axs[idx].grid(True)

    plt.savefig(f'../reports/figures/{folder}/{borough}_pred_vs_actual.jpg', dpi=300)
    plt.show()



def plot_demand_by_week_borough(df_1, df_2, df_3, df_4, y_pred, borough):
    # Filter out the data for the specified borough
    borough_data_1 = df_1.loc[df_1[f'start_borough_{borough}'] == 1].copy()
    borough_data_2 = df_2.loc[df_2[f'start_borough_{borough}'] == 1].copy()
    borough_data_3 = df_3.loc[df_3[f'start_borough_{borough}'] == 1].copy()
    borough_data_4 = df_4.loc[df_4[f'start_borough_{borough}'] == 1].copy()

    # Add columns for the date, hour, and week for training data
    borough_data_1.loc[:, 'date'] = borough_data_1['start_date_hour'].dt.date
    borough_data_1.loc[:, 'week_of_year'] = borough_data_1['start_date_hour'].dt.isocalendar().week
    borough_data_1.loc[:, 'hour_of_week'] = borough_data_1['day_of_week'] * 24 + borough_data_1['hour']

    borough_data_2.loc[:, 'date'] = borough_data_2['start_date_hour'].dt.date
    borough_data_2.loc[:, 'week_of_year'] = borough_data_2['start_date_hour'].dt.isocalendar().week
    borough_data_2.loc[:, 'hour_of_week'] = borough_data_2['day_of_week'] * 24 + borough_data_2['hour']

    borough_data_3.loc[:, 'date'] = borough_data_3['start_date_hour'].dt.date
    borough_data_3.loc[:, 'week_of_year'] = borough_data_3['start_date_hour'].dt.isocalendar().week
    borough_data_3.loc[:, 'hour_of_week'] = borough_data_3['day_of_week'] * 24 + borough_data_3['hour']

    borough_data_4.loc[:, 'date'] = borough_data_4['start_date_hour'].dt.date
    borough_data_4.loc[:, 'week_of_year'] = borough_data_4['start_date_hour'].dt.isocalendar().week
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

    plt.show()

    return borough_data_1, borough_data_2, borough_data_3, borough_data_4



def get_enornous_entrie(y_test, y_pred, x_test):
    # Calculate absolute error for each prediction
    errors = abs(y_pred - y_test)

    # Create a DataFrame with the actual, predicted values, and errors
    df_errors = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': errors
    }).sort_values(by='Error', ascending=False)

    # Add additional columns to df_errors_entries
    df_errors['day_of_week'] = x_test['day_of_week']
    df_errors['hour'] = x_test['hour']
    df_errors['month'] = x_test['month']
    df_errors['bank_holiday'] = x_test['bank_holiday']
    df_errors['start_borough_Westminster'] = x_test['start_borough_Westminster']

    return df_errors



def hyper_param_tuning(x_train, y_train, n_iter, n_splits):
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': randint(100, 1000),  # Number of trees in the random forest
        'max_depth': randint(5, 20),  # Maximum depth of the tree
        'min_samples_split': randint(2, 10),  # Minimum number of samples required to split a node
        'min_samples_leaf': randint(1, 10)  # Minimum number of samples required at each leaf node
    }

    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Create a TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Perform Randomized Search with 10 iterations
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=n_iter, cv=tscv, random_state=42)

    # Fit the random search on the train data
    random_search.fit(x_train, y_train)

    return random_search



def visualize_random_search_rf(random_search):
    # Define the features and their corresponding parameter ranges
    features = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    param_ranges = [(100, 1000), (5, 20), (2, 10), (1, 10)]

    # Get the parameter combinations, scores, and fit times
    params = random_search.cv_results_['params']
    scores = random_search.cv_results_['mean_test_score']
    fit_times = random_search.cv_results_['mean_fit_time']

    # Create subplots for each feature
    fig, axs = plt.subplots(nrows=len(features), ncols=2, figsize=(8, 4*len(features)))

    # Plot feature vs score and feature vs training time for each feature
    for i, feature in enumerate(features):
        feature_values = [param[feature] for param in params]

        # Plot feature vs score
        ax1 = axs[i, 0]
        ax1.plot(feature_values, scores, marker='o')
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Score')
        ax1.set_title(f'{feature} vs Score')
        ax1.grid(True)

        # Plot feature vs training time
        ax2 = axs[i, 1]
        ax2.plot(feature_values, fit_times, marker='o', color='orange')
        ax2.set_xlabel(feature)
        ax2.set_ylabel('Training Time (s)')
        ax2.set_title(f'{feature} vs Training Time')
        ax2.grid(True)

    plt.tight_layout()
    plt.show()




def hyper_param_tuning_gb(x_train, y_train, n_iter, n_splits):
    param_grid = {
        'n_estimators': randint(100, 1000),  # Number of boosting stages to perform
        'max_depth': randint(3, 10),  # Maximum depth of the individual regression estimators
        'min_samples_split': randint(2, 10),  # Minimum number of samples required to split a node
        'min_samples_leaf': randint(1, 10),  # Minimum number of samples required at each leaf node
        'learning_rate': [0.01, 0.05, 0.1, 0.2]  # Learning rate
    }

    # Initialize the Gradient Boosting Regressor
    gb = GradientBoostingRegressor(random_state=42)

    # Create a TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Perform Randomized Search with specified iterations
    random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_grid, n_iter=n_iter, cv=tscv, random_state=42)

    # Fit the random search on the train data
    random_search.fit(x_train, y_train)

    return random_search