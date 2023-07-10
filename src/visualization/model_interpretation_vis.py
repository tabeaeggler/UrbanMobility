import numpy as np
from alibi.explainers import ALE, plot_ale
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap
import pickle
from PIL import Image
from IPython.display import display


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


def ale_explainer(model, feature_names, data):

    # ALE
    ale = ALE(model.predict, feature_names=feature_names, target_names=['demand'])

    # Convert DataFrame to numpy array, ensure columns match feature_names
    x_daily_test_np = data.to_numpy()

    # Generate the explanation with numpy array
    ale_exp = ale.explain(x_daily_test_np)

    return ale_exp


def permutation_tree(model, x_data, y_data):

    # Calculate permutation
    permutation = permutation_importance(model, x_data, y_data, n_repeats=5, random_state=42, n_jobs=-1)
    perm_importance = permutation.importances_mean

    # Combine feature names and importances into a list of tuples
    feature_importances = [(name, importance) for name, importance in zip(x_data.columns, perm_importance)]

    # Sort the feature importances by the importance score
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    return feature_importances


def create_shap(model, x_data, filename_expl, filename_val):
    # Create the explainer
    explainer = shap.TreeExplainer(model)
    expected_value = explainer.expected_value

    # Compute SHAP values
    shap_values = explainer(x_data)

    # save explainer and values
    pickle.dump(explainer, open(filename_expl, 'wb'))
    pickle.dump(shap_values, open(filename_val, 'wb'))

    return shap_values



def plot_shap_bar_and_beeswarm(shap_values, filename_bar_plot, filename_beeswarm_plot, filename_final):

    # Plot the global bar plot
    plt.title("SHAP Bar Plot - Global Feature Importance")
    shap.plots.bar(shap_values, max_display=30, show=False)
    plt.savefig(filename_bar_plot, bbox_inches='tight')
    plt.clf()

    # Plot the global beeswarm plot
    plt.title("SHAP Beeswarm Plot - Feature Impact on Output")
    shap.plots.beeswarm(shap_values, max_display=30, show=False)
    plt.savefig(filename_beeswarm_plot, bbox_inches='tight')
    plt.clf()

    # Open the two images
    image_bar = Image.open(filename_bar_plot)
    image_beeswarm = Image.open(filename_beeswarm_plot)

    # Determine the new dimensions: here, we choose the smaller width and height
    new_size = (min(image_bar.size[0], image_beeswarm.size[0]), min(image_bar.size[1], image_beeswarm.size[1]))

    # Resize images while keeping the aspect ratio
    image_bar.thumbnail(new_size, Image.ANTIALIAS)
    image_beeswarm.thumbnail(new_size, Image.ANTIALIAS)

    # Create a new image with the right size and white background
    new_image = Image.new('RGB', (new_size[0] * 2, new_size[1]), 'white')

    # Paste the images side by side
    new_image.paste(image_bar, (0, 0))
    new_image.paste(image_beeswarm, (new_size[0], 0))

    # Save the final image
    new_image.save(filename_final)
    display(new_image)



def plot_boroughs_bar(shap_values, x_data, boroughs, global_min, global_max, filename_bar_plots, filename_final_boroughs):

    # Get the indices in the shap_values where each borough is true
    borough_indices = {borough: np.where(x_data['start_borough_' + borough] == 1)[0] for borough in boroughs}

    # Calculate the global min and max SHAP values
    global_min = 0
    global_max = 800

    for i, (borough, indices) in enumerate(borough_indices.items()):
        # Select only the rows for this borough
        borough_shap_values = shap_values[indices]
        
        # Plot the bar plot for this borough
        shap.plots.bar(borough_shap_values, max_display=30, show=False)
        plt.title(f"SHAP Bar Plot - Borough {borough}")

        # Set x limits
        plt.xlim(global_min, global_max)
        
        # Save the plot as a PNG
        plt.savefig(f'{filename_bar_plots}{borough}_shap_plot.png', bbox_inches='tight')
        
        # Clear the current plot to make room for the next one
        plt.clf()

    # Open all the images
    images = [Image.open(f'{filename_bar_plots}{borough}_shap_plot.png') for borough in boroughs]

    # Number of images per row
    n_cols = 3

    # Calculate the number of rows
    n_rows = len(boroughs) // n_cols
    n_rows += len(boroughs) % n_cols

    # Calculate the max width and height
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new image with the right size and white background
    new_image = Image.new('RGB', (max_width * n_cols, max_height * n_rows), 'white')

    # Paste each image next to each other
    x_offset = 0
    y_offset = 0
    for i, image in enumerate(images):
        new_image.paste(image, (x_offset, y_offset))
        if (i+1) % n_cols == 0:  # Move to next row after every n_cols images
            y_offset += max_height
            x_offset = 0
        else:
            x_offset += max_width

    # Save the final image
    new_image.save(filename_final_boroughs)
    display(new_image)


