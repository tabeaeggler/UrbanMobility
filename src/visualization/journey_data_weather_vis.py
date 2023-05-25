import matplotlib.pyplot as plt
import seaborn as sns


def correlation_plot(df, title):

    # calculate the correlation matrix
    correlation_matrix = df.corr()

    # create the correlation plot
    sns.set(style="white")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(title)

    return plt