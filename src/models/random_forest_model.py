import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

    rf = RandomForestRegressor(n_estimators=400, random_state=42)

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


def evaluation_actual_vs_predicted(y_test, y_pred, x_test, borough):
    # Create a copy of the test DataFrame and add predicted demand
    evaluation_df = x_test.copy()
    evaluation_df['actual_demand'] = y_test
    evaluation_df['predicted_demand'] = y_pred

    evaluation_df['start_date_hour'] = pd.to_datetime(evaluation_df['start_date_hour'])

    # Create date, day of week, week of year and hour of week columns
    evaluation_df['date'] = evaluation_df['start_date_hour'].dt.date
    evaluation_df['day_of_week'] = evaluation_df['start_date_hour'].dt.dayofweek
    evaluation_df['week_of_year'] = evaluation_df['start_date_hour'].dt.isocalendar().week
    evaluation_df['hour_of_week'] = evaluation_df['day_of_week'] * 24 + evaluation_df['start_date_hour'].dt.hour
    
    # Aggregate by week and hour of the week
    evaluation_weekly_hourly = evaluation_df.groupby(['week_of_year', 'hour_of_week']).agg({'actual_demand': 'sum', 'predicted_demand': 'sum'}).reset_index()

    # Create subplots for each week
    fig, axs = plt.subplots(nrows=52, figsize=(30, 90), constrained_layout=True, sharey='row')

    # Plot each week's data for actual and predicted demand
    for idx, week in enumerate(evaluation_weekly_hourly['week_of_year'].unique()):
        # Extract data for this week
        week_data = evaluation_weekly_hourly[evaluation_weekly_hourly['week_of_year'] == week]
        
        # Plot demand for this week
        axs[idx].plot(week_data['hour_of_week'], week_data['predicted_demand'], color='red', label='Predicted demand')
        axs[idx].plot(week_data['hour_of_week'], week_data['actual_demand'], color='blue', label='Actual demand')

        axs[idx].set_title(f'Week {week} - {borough} (Actual vs. Predicted)')
        axs[idx].set_ylabel('Demand')
        axs[idx].set_xticks(np.arange(0, 168, 24))  # Set x-axis ticks every 24 hours
        axs[idx].legend(loc="upper right")
        axs[idx].grid(True)

    plt.savefig(f'../reports/figures/model_random_forest/{borough}_pred_vs_actual', dpi=300)
    plt.show()




def get_enornous_entrie(y_test, y_pred, x_test):
    # calc absolute error for each prediction
    errors = abs(y_pred - y_test)

    # create a DataFrame with the actual, predicted values and errors
    df_errors = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': errors
    }).sort_values(by='Error', ascending=False)

    # retrieve the corresponding rows from the original dataset
    top_error_indices = df_errors.head(20).index
    df_errors_entries = x_test.loc[top_error_indices]

    return (df_errors, df_errors_entries)



def hyper_param_tuning(x_train, y_train):
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
    tscv = TimeSeriesSplit(n_splits=3)

    # Perform Randomized Search with 10 iterations
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=tscv, random_state=42)

    # Fit the random search on the train data
    random_search.fit(x_train, y_train)

    return pd.DataFrame(random_search.cv_results_)



