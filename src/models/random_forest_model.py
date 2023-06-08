import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_demand_hourly(journey_train, journey_test, standardise = False):
    # aggregate demand for each borough by summing
    journey_train['demand'] = 1
    journey_test['demand'] = 1

    # caet a list of all column names except 'demand' and 'start_date'
    column_names = [col for col in journey_train.columns]

    # create a dictionary with all features, first -> always keep values of first element
    aggregate_functions = {col: 'first' for col in column_names}

    # add an entry for the 'demand' column with 'sum' as the aggregate function
    aggregate_functions['demand'] = 'sum'

    # round down the datetime to the nearest hour
    journey_train['start_date_hour'] = journey_train['start_date'].dt.floor('H')
    journey_test['start_date_hour'] = journey_test['start_date'].dt.floor('H')

    # perform the groupby operation
    journey_train_hourly = journey_train.groupby(['start_date_hour', 'start_borough']).agg(aggregate_functions)
    journey_test_hourly = journey_test.groupby(['start_date_hour', 'start_borough']).agg(aggregate_functions)

    # one hot encoding start_borough
    journey_train_hourly = pd.get_dummies(journey_train_hourly, columns=['start_borough'])
    journey_test_hourly = pd.get_dummies(journey_test_hourly, columns=['start_borough'])

    journey_train_hourly = journey_train_hourly.reset_index()
    journey_test_hourly = journey_test_hourly.reset_index()

    if standardise:
        # standardise the demand per dock
        journey_train_hourly['demand_per_dock'] = journey_train_hourly['demand'] / journey_train_hourly['bike_docks_counts']
        journey_test_hourly['demand_per_dock'] = journey_test_hourly['demand'] / journey_test_hourly['bike_docks_counts']

        # create the target variables
        y_train = journey_train_hourly['demand_per_dock']
        y_test = journey_test_hourly['demand_per_dock']

        cols_to_remove = ['rental_id', 'end_date', 'end_borough', 'start_date', 'end_station_name', 'start_station_name', 'demand', 'borough', 'borough_code', 'year', 'start_borough', 'start_date_hour', 'demand_per_dock']

    else:
        # create the target variables
        y_train = journey_train_hourly['demand']
        y_test = journey_test_hourly['demand']

        cols_to_remove = ['rental_id', 'end_date', 'end_borough', 'start_date', 'end_station_name', 'start_station_name', 'demand', 'borough', 'borough_code', 'year', 'start_borough', 'start_date_hour']
    
    # create the predictor variables
    x_train = journey_train_hourly.drop(columns=cols_to_remove)
    x_test = journey_test_hourly.drop(columns=cols_to_remove)

    return (x_train, y_train, x_test, y_test, journey_train_hourly)


def random_forest_fit_pred(x_train, y_train, x_test):

    rf = RandomForestRegressor(n_estimators=2, random_state=42)

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    return (rf, y_pred)

def gradient_boosting_fit_pred(x_train, y_train, x_test):
    from sklearn.ensemble import GradientBoostingRegressor

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

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

def evaluation_vis(y_test, y_pred, x_train):

    fig, axs = plt.subplots(3, 2, figsize=(20, 18))

    # Plot 1: Actual vs Predicted
    axs[0, 0].scatter(y_test, y_pred, alpha=0.3)
    axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    axs[0, 0].set_xlabel('Actual')
    axs[0, 0].set_ylabel('Predicted')
    axs[0, 0].set_title('Actual vs Predicted')

    # Plot 2: Residuals vs Predicted
    residuals = y_test - y_pred
    axs[0, 1].scatter(y_pred, residuals, alpha=0.3)
    axs[0, 1].axhline(0, color='k', linestyle='--', lw=4)
    axs[0, 1].set_xlabel('Predicted')
    axs[0, 1].set_ylabel('Residuals')
    axs[0, 1].set_title('Residuals vs Predicted')

    # Plot 3: Histogram of Residuals
    axs[1, 0].hist(residuals, bins=30)
    axs[1, 0].axvline(0, color='k', linestyle='--', lw=4)
    axs[1, 0].set_xlabel('Residuals')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Histogram of Residuals')

    # Plot 4: Boxplot of demand by hour
    sns.boxplot(x='hour', y='demand', data=x_train, ax=axs[1, 1])
    axs[1, 1].set_title('Boxplot of demand by hour')
    axs[1, 1].set_xlabel('Hour of the day')
    axs[1, 1].set_ylabel('Demand')
    axs[1, 1].set_xticks(range(24))

    # Plot 5: Actual vs Predicted Comparison
    axs[2, 0].remove() # We remove this plot as we want the final plot to span two columns
    axs[2, 1].plot(y_pred[0:1000],'r', label='Predicted')
    axs[2, 1].plot(y_test[0:1000].values, label='Actual')
    axs[2, 1].set_xlabel('Time')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].set_title('Actual vs Predicted Comparison')
    axs[2, 1].legend()

    # To make the last plot span two columns
    gs = axs[2, 1].get_gridspec()
    axbig = fig.add_subplot(gs[2, :])

    axbig.plot(y_pred[0:1000],'r', label='Predicted')
    axbig.plot(y_test[0:1000].values, label='Actual')
    axbig.set_xlabel('Time')
    axbig.set_ylabel('Value')
    axbig.set_title('Actual vs Predicted Comparison')
    axbig.legend()

    plt.tight_layout()
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


def hyper_param_tuning():
    # Hyperparam tuning

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': randint(100, 1000),  # Number of trees in the random forest
        'max_features': ['auto', 'sqrt'],  # Number of features to consider at each split
        'max_depth': randint(5, 20),  # Maximum depth of the tree
        'min_samples_split': randint(2, 10),  # Minimum number of samples required to split a node
        'min_samples_leaf': randint(1, 10)  # Minimum number of samples required at each leaf node
    }

    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Perform Randomized Search with 10 iterations
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)

    # Fit the random search on the train data
    random_search.fit(x_train, y_train)

    # Get the best model and its hyperparameters
    best_rf = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Best Hyperparameters:", best_params)

    # Predict on the test set using the best model
    y_pred_best = best_rf.predict(x_test)

    # Calculate updated evaluation metrics
    rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)
    evs_best = explained_variance_score(y_test, y_pred_best)

    print('Best Model RMSE:', rmse_best)
    print('Best Model MAE:', mae_best)
    print('Best Model R2 Score:', r2_best)
    print('Best Model Explained Variance Score:', evs_best)



