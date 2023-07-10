import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def evaluation_vis(y_test, y_pred):
    """
    Visualize the model's performance with scatter plots of actual vs predicted values and residuals, 
    as well as a histogram of residuals.

    Args:
    y_test (numpy.ndarray): The actual output values.
    y_pred (numpy.ndarray): The predicted output values by the model.
    """
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
    """
    Evaluate model performance visually by comparing actual and predicted demand in a borough for every hour of every week.

    Args:
    y_test (numpy.ndarray): The actual output values.
    y_pred (numpy.ndarray): The predicted output values by the model.
    x_test (pandas.DataFrame): The testing set data.
    borough (str): The name of the borough.
    folder (str): The name of the folder where the plot will be saved.
    """
        
    # create a copy of the test DataFrame and add predicted demand
    evaluation_df = x_test.copy()
    evaluation_df['actual_demand'] = y_test
    evaluation_df['predicted_demand'] = y_pred

    # filter the data for the given borough
    if borough != 'All_Boroughs':
        evaluation_df = evaluation_df[evaluation_df[f'start_borough_{borough}'] == 1]

    # create date, day of week, week of year and hour of week columns
    evaluation_df['start_date'] = pd.to_datetime(evaluation_df['start_date'])
    evaluation_df['date'] = evaluation_df['start_date'].dt.date
    evaluation_df['day_of_week'] = evaluation_df['start_date'].dt.dayofweek
    evaluation_df['week_of_year'] = evaluation_df['start_date'].dt.isocalendar().week
    evaluation_df['hour_of_week'] = evaluation_df['day_of_week'] * 24 + evaluation_df['start_date'].dt.hour
        
    # aggregate by week and hour of the week
    evaluation_weekly_hourly = evaluation_df.groupby(['week_of_year', 'hour_of_week']).agg({'actual_demand': 'sum', 'predicted_demand': 'sum'}).reset_index()

    # create subplots for each week
    fig, axs = plt.subplots(nrows=52, figsize=(30, 70), constrained_layout=True, sharey='row')

    # plot each week's data for actual and predicted demand
    for idx, week in enumerate(evaluation_weekly_hourly['week_of_year'].unique()):
        # extract data for this week
        week_data = evaluation_weekly_hourly[evaluation_weekly_hourly['week_of_year'] == week]
            
        # plot demand for this week
        axs[idx].plot(week_data['hour_of_week'], week_data['predicted_demand'], color='red', label='Predicted demand')
        axs[idx].plot(week_data['hour_of_week'], week_data['actual_demand'], color='blue', label='Actual demand (test, 2019)')

        axs[idx].set_title(f'Week {week} - {borough} (Actual vs. Predicted)')
        axs[idx].set_ylabel('Demand')
        axs[idx].set_xticks(np.arange(0, 168, 24))  # Set x-axis ticks every 24 hours
        axs[idx].legend(loc="upper right")
        axs[idx].grid(True)

    plt.savefig(f'../reports/figures/{folder}/{borough}_pred_vs_actual.jpg', dpi=300)


def visualize_random_search(random_search, features, title):
    """
    Visualize the relationship between the parameters and performance score for each feature in a model used for random search.

    Args:
    random_search (sklearn.RandomizedSearchCV): The result of fitting RandomizedSearchCV.
    features (list of str): A list of the names of the features.
    title (str): The title of the plot.
    """

    # get the parameter combinations, scores, and fit times
    params = random_search.cv_results_['params']
    scores = random_search.cv_results_['mean_test_score']

    # get the sorted indices
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
    """
    Calculate and return a dataframe of absolute errors for each prediction, along with relevant details from the test dataset.
    
    Args:
    y_test (array-like): Actual target values for the test set.
    y_pred (array-like): Predicted target values for the test set.
    x_test (DataFrame): Test features data.
    
    Returns:
    df_errors (DataFrame): Dataframe of actual, predicted values, error, day_of_week, hour, month, bank_holiday, and start_borough_Westminster.
    The dataframe is sorted by 'Error' in descending order.
    """

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



def get_feature_importance(tree_model, x_train):
    """
    Function to compute and return gini feature importance for tree models.
    
    Args:
    tree_model: A trained model that has the attribute "feature_importances_".
    x_train (DataFrame): The training features data.
    
    Returns:
    feature_importances (DataFrame): DataFrame of features and their corresponding importance. Dataframe is sorted by 'Importance' in descending order.
    """
    importances = tree_model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)

    return feature_importances



def loss_curve(history, title):
    """
    Function to plot the training and validation loss.

    Args:
    history (History): A history object obtained from the fit method of a keras Model instance.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['loss'], 'b', linewidth=2)
    plt.plot(history.history['val_loss'], 'orange', linewidth=2)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=10)
    plt.title(title, fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    


def evaluation_metrics(y_test, y_pred):

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return (rmse, mae, r2)
