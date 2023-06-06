import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def correlation_plot(df, title):

    """
    Creates a correlation plot (heatmap) for the given DataFrame, showing the 
    correlation coefficients between pairs of variables.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data to plot. 
    title (str): The title of the plot.

    Returns:
    matplotlib.pyplot: A correlation plot (heatmap) object.
    """

    # calculate the correlation matrix
    correlation_matrix = df.corr()

    # create the correlation plot
    sns.set(style="white")
    plt.figure(figsize=(14, 10))
    sns.heatmap(np.around(correlation_matrix, decimals=2), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(title)

    return plt
