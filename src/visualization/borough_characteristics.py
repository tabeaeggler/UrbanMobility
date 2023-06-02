import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def scatterplot_borough_charactersitics(borough_df, demand_var, title):


    # Define a function to find outliers
    def find_outliers(df, column1, column2):
        df = df.copy()  # Make a copy to avoid modifying original df
        df['Z_score1'] = zscore(df[column1])
        df['Z_score2'] = zscore(df[column2])
        outliers_df = df[(abs(df['Z_score1']) > 1.2) | (abs(df['Z_score2']) > 1.2)]
        return outliers_df

    # List of columns to exclude
    exclude_columns = ['borough', 'borough_code', 'demand_count_2019_start_borough', 
                    'demand_count_2019_end_borough', 'demand_stand_count_2019_start_borough', 
                    'demand_stand_count_2019_end_borough']

    # Get a list of all other columns in the dataframe
    variables = [col for col in borough_df.columns if col not in exclude_columns]

    # Calculate number of rows needed for all subplots
    n_rows = len(variables) // 5
    if len(variables) % 5:
        n_rows += 1

    # Create subplots
    fig, axs = plt.subplots(n_rows, 5, figsize=(40, 6*n_rows), sharey=True)

    # Flatten axs object if needed
    if n_rows > 1:
        axs = axs.ravel()

    # Dictionary to store x-ranges for prefixes
    prefix_xrange = {}

    # Iterate through each subplot and create a scatterplot
    for i, var in enumerate(variables):
        prefix = var.split('_')[0]  # Get prefix of variable before the first underscore
        if prefix not in prefix_xrange:
            # Compute x-range for this prefix
            min_x = borough_df[borough_df.columns[borough_df.columns.str.startswith(prefix)]].min().min()
            max_x = borough_df[borough_df.columns[borough_df.columns.str.startswith(prefix)]].max().max()
            prefix_xrange[prefix] = (min_x, max_x)

        sns.scatterplot(x=var, y=demand_var, data=borough_df, ax=axs[i], color='purple', s=100)
        axs[i].set_xlabel(var, fontsize=20)
        axs[i].set_ylabel('Demand Standardized', fontsize=20)
        axs[i].tick_params(axis='both', which='major', labelsize=16)
        axs[i].set_xlim(prefix_xrange[prefix])  # Set x-axis limits

        outliers_df = find_outliers(borough_df, var, demand_var)
        for k in outliers_df.index:
            axs[i].annotate(outliers_df.loc[k, 'borough'], (outliers_df.loc[k, var], outliers_df.loc[k, demand_var]))

    # Remove empty subplots
    for i in range(len(variables), n_rows*5):
        fig.delaxes(axs[i])

    plt.suptitle(title, fontsize=20, y=1)
    plt.tight_layout()

    return plt



def heatmap_borough_charactersitics(borough_df):
    # calculate the correlation matrix
    correlation_matrix = borough_df.corr()

    # select rows of the correlation matrix
    correlation_matrix_subset = correlation_matrix.loc[['demand_count_2019_start_borough', 'demand_stand_count_2019_start_borough'], :]

    # transpose the subset of the correlation matrix
    correlation_matrix_subset = correlation_matrix_subset.transpose()

    # create the correlation plot
    sns.set(style="white")
    plt.figure(figsize=(14, 30))
    sns.heatmap(correlation_matrix_subset, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation matrix for all variables against demands")

    return plt
