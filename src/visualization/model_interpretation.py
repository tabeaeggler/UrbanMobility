import numpy as np
from alibi.explainers import ALE, plot_ale
import matplotlib.pyplot as plt

def ale_plot(x_test, ale_exp, selected_features, height, spacing, title, large_scale_index):

    # Convert the feature names to indices
    selected_feature_indices = [list(x_test.columns).index(name) for name in selected_features]

    nrows = int(np.ceil(len(selected_features) / 6))
    ncols = 6

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, height))

    # Flatten the axes array in case it's 2D (when you have more than one row)
    axs = axs.flatten()

    for ax, feature_index in zip(axs, selected_feature_indices):
        plot_ale(ale_exp, features=[feature_index], ax=ax)

        # set another y-axis scale for hour feature
        if feature_index in large_scale_index:
            ax.set_ylabel('ALE (large scale!)')
            ax.set_ylim([-400,900])
        else:
            ax.set_ylim([-180,180])

    # Disable remaining axes
    for ax in axs[len(selected_features):]:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=spacing)
    plt.suptitle(f'ALE plot for {title}')
    plt.show()
