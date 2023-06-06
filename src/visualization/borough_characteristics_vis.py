import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


def add_yearly_demands(borough_df, journey_df):
    """
    Adds yearly demand and demand standardized by bike station counts to borough DataFrame.

    Args:
        journey_19_df (DataFrame): DataFrame containing journey data for 2019.
        borough_df (DataFrame): DataFrame containing borough data.

    Returns:
        DataFrame: The updated borough DataFrame with added yearly demand and standardized demand.
    """
    # add yearly demand and demand standardised by bike station counts
    borough_df = borough_df.set_index('borough', drop=False)
    demand_start = journey_df.groupby('start_borough')['start_station_name'].count()
    demand_end = journey_df.groupby('end_borough')['start_station_name'].count()
    borough_df['demand_19_start_borough'] = demand_start
    borough_df['demand_19_end_borough'] = demand_end

    borough_df.reset_index(drop=True, inplace=True)

    # calculate standardized demand
    borough_df['demand_stand_19_start_borough'] = borough_df['demand_19_start_borough'] / borough_df['bike_docks_counts']
    borough_df['demand_stand_19_end_borough'] = borough_df['demand_19_end_borough'] / borough_df['bike_docks_counts']

    return borough_df



def scatterplot_borough_charactersitics(borough_df, demand_var, title):
    """
    Generates scatter plots of borough characteristics against demand with same x-range for same feature prefix.

    Args:
        borough_df (DataFrame): DataFrame containing borough characteristics.
        demand_var (str): Variable indicating the demand.
        title (str): Title of the plot.

    Returns:
        matplotlib.pyplot: Scatter plot object.
    """

    # define a function to find outliers
    def find_outliers(df, column1, column2):
        df = df.copy() 
        df['Z_score1'] = zscore(df[column1])
        df['Z_score2'] = zscore(df[column2])
        outliers_df = df[(abs(df['Z_score1']) > 1.2) | (abs(df['Z_score2']) > 1.2)]
        return outliers_df

    # list of columns to exclude
    exclude_columns = ['borough', 'borough_code', 'demand_19_start_borough', 
                    'demand_19_end_borough', 'demand_stand_19_start_borough', 
                    'demand_stand_19_end_borough']

    variables = [col for col in borough_df.columns if col not in exclude_columns]

    # calculate number of rows needed for all subplots
    n_rows = len(variables) // 5
    if len(variables) % 5:
        n_rows += 1

    # create subplots
    fig, axs = plt.subplots(n_rows, 5, figsize=(40, 6*n_rows), sharey=True)

    # flatten axs object if needed
    if n_rows > 1:
        axs = axs.ravel()

    # dictionary to store x-ranges for prefixes
    prefix_xrange = {}

    # iterate through each subplot and create a scatterplot
    for i, var in enumerate(variables):
        prefix = var.split('_')[0]
        if prefix not in prefix_xrange:
            # compute x-range for this prefix
            min_x = borough_df[borough_df.columns[borough_df.columns.str.startswith(prefix)]].min().min()
            max_x = borough_df[borough_df.columns[borough_df.columns.str.startswith(prefix)]].max().max()
            prefix_xrange[prefix] = (min_x, max_x)

        sns.scatterplot(x=var, y=demand_var, data=borough_df, ax=axs[i], color='purple', s=100)
        axs[i].set_xlabel(var, fontsize=20)
        axs[i].set_ylabel('Demand', fontsize=20)
        axs[i].tick_params(axis='both', which='major', labelsize=16)
        axs[i].set_xlim(prefix_xrange[prefix])  # Set x-axis limits

        outliers_df = find_outliers(borough_df, var, demand_var)
        for k in outliers_df.index:
            axs[i].annotate(outliers_df.loc[k, 'borough'], (outliers_df.loc[k, var], outliers_df.loc[k, demand_var]))

    # Remove empty subplots
    for i in range(len(variables), n_rows*5):
        fig.delaxes(axs[i])

    plt.suptitle(title, fontsize=30, y=1)
    plt.tight_layout()

    return plt



def heatmap_borough_charactersitics(borough_df):
    """
    Generates a heatmap of correlation matrix of all borough characteristics.

    Args:
        borough_df (DataFrame): DataFrame containing borough characteristics.

    Returns:
        matplotlib.pyplot: Heatmap plot object.
    """
    # create correlation matrix
    correlation_matrix = borough_df.corr()

    correlation_matrix_subset = correlation_matrix.loc[['demand_19_start_borough', 'demand_19_end_borough' , 'demand_stand_19_start_borough', 'demand_stand_19_end_borough'], :]
    correlation_matrix_subset = correlation_matrix_subset.transpose()

    # create the correlation plot
    sns.set(style="white")
    plt.figure(figsize=(14, 30))
    sns.heatmap(correlation_matrix_subset, annot=True, cmap="coolwarm", linewidths=0.5, cbar=False)
    plt.title("Correlation Matrix of all Features against Demands", fontsize=15)

    return plt

