import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error



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


def evaluation_actual_vs_predicted_hourly(y_test, y_pred, x_test, borough, folder):
    # Create a copy of the test DataFrame and add predicted demand
    evaluation_df = x_test.copy()
    evaluation_df['actual_demand'] = y_test
    evaluation_df['predicted_demand'] = y_pred

    # Filter the data for the given borough
    if borough != 'All_Boroughs':
        evaluation_df = evaluation_df[evaluation_df[f'start_borough_{borough}'] == 1]

    # Create date, day of week, week of year and hour of week columns
    evaluation_df['start_date'] = pd.to_datetime(evaluation_df['start_date'])
    evaluation_df['date'] = evaluation_df['start_date'].dt.date
    evaluation_df['day_of_week'] = evaluation_df['start_date'].dt.dayofweek
    evaluation_df['week_of_year'] = evaluation_df['start_date'].dt.isocalendar().week
    evaluation_df['hour_of_week'] = evaluation_df['day_of_week'] * 24 + evaluation_df['start_date'].dt.hour
        
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
    #plt.show()


def visualize_random_search(random_search, features, title):
    # get the parameter combinations, scores, and fit times
    params = random_search.cv_results_['params']
    scores = random_search.cv_results_['mean_test_score']

    # get the indices that would sort the scores
    sorted_indices = np.argsort(scores)

    # create subplots for each feature
    fig, axs = plt.subplots(nrows=1, ncols=len(features), figsize=(4*len(features), 4))

    # plot feature vs score for each feature
    for i, feature in enumerate(features):
        feature_values = [param[feature] for param in params]
        sorted_feature_values = [feature_values[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]


        ax = axs[i]
        ax.scatter(sorted_feature_values, sorted_scores, marker='o')
        ax.set_xlabel(feature)
        ax.set_ylabel('Score')
        ax.grid(True)

    plt.tight_layout()
    fig.suptitle(title, y=1.05)
    plt.show()


def get_enornous_entrie(y_test, y_pred, x_test):
    # calculate absolute error for each prediction
    errors = abs(y_pred - y_test)

    # create a DataFrame with the actual, predicted values, and errors
    df_errors = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': errors
    }).sort_values(by='Error', ascending=False)

    # add additional columns to df_errors_entries
    df_errors['day_of_week'] = x_test['day_of_week']
    df_errors['hour'] = x_test['hour']
    df_errors['month'] = x_test['month']
    df_errors['bank_holiday'] = x_test['bank_holiday']
    df_errors['start_borough_Westminster'] = x_test['start_borough_Westminster']

    return df_errors


def evaluation_metrics(y_test, y_pred):

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return (rmse, mae, r2)



def get_feature_importance(rf, x_train):

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)

    return feature_importances