import numpy as np
from alibi.explainers import ALE, plot_ale
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap
import pickle
from PIL import Image
from IPython.display import display
import seaborn as sns



def feature_importance_stacked_barplot(df, fig_height=5):

    colors = sns.color_palette('deep')[0:5]

    # Transpose DataFrame for easier plotting
    df = df.set_index('Features').T

    # Create stacked horizontal bar chart
    ax = df.plot(kind='barh', stacked=True, color=colors, edgecolor='black', linewidth=1.5, figsize=(18, fig_height))

    # Setting labels
    plt.xlabel('Importance')
    plt.ylabel('Models')

    # Setting title
    plt.title('Impurity Feature Importance Attribution')

    # Add percentages on top of bars
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        if width > 2:  # Only annotate when the width is greater than 5%
            ax.annotate(f"{width:.1f}%", xy=(left+width/2, bottom+height/2), 
                        xytext=(0,0), textcoords='offset points', ha='center', va='center')
            
    # Move the legend to the right side of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'../reports/figures/model_interpretation/feature_importance_cat', dpi=300)
    plt.show()

    

def top_8_features_plot(df_impurity, model):
    other_feature = ['day_of_week', 'is_weekend', 'day_of_month', 'day_of_year', 'month', 'season', 'hour', 'part_of_day', 'bank_holiday', 'tempmax', 'tempmin', 'temp', 'feelslike', 'humidity', 'dew', 'precip', 'windgust', 'windspeed', 'cloudcover', 'visibility', 'uvindex', 'daylight_hours', 'bike_station_counts', 'bike_docks_counts', 'demand_ma_3h', 'demand_ma_8h', 'demand_ma_24h', 'demand_lag_1h', 'demand_lag_8h', 'demand_lag_24h', 'demand_lag_1w']
    borough_features = [feature for feature in df_impurity['Feature'] if feature not in other_feature]


    # Extract top 8 features
    top_8_features = df_impurity.nlargest(8, 'Importance')

    # Extract top 8 borough features
    borough_df = df_impurity[df_impurity['Feature'].isin(borough_features)]
    top_8_borough_features = borough_df.nlargest(8, 'Importance')

    # Left plot: Top 8 features
    fig, axs = plt.subplots(1, 2, figsize=(20, 3.4))
    axs[0].barh(top_8_features['Feature'], top_8_features['Importance'], color='steelblue')
    axs[0].invert_yaxis()  # Invert y-axis to display the highest value at the top
    axs[0].set_xlabel('Importance')
    axs[0].set_xlim([0, 0.8])
    axs[0].set_title('Top 8 Features')

    # Right plot: Top 8 borough features
    axs[1].barh(top_8_borough_features['Feature'], top_8_borough_features['Importance'], color='sandybrown')
    axs[1].invert_yaxis()  # Invert y-axis to display the highest value at the top
    axs[1].set_xlabel('Importance')
    axs[1].set_xlim([0, 0.8])
    axs[1].set_title('Top 8 Borough Features')

    fig.suptitle(f'Impurity Feature Importance, Model {model}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f'../reports/figures/model_interpretation/top_features_{model}', dpi=300)
    plt.show()



def ale_plot(x_test, ale_exp_1, ale_exp_2, selected_features, height, spacing, title, large_scale_index):

    # Convert the feature names to indices
    selected_feature_indices = [list(x_test.columns).index(name) for name in selected_features]

    nrows = int(np.ceil(len(selected_features) / 6))
    ncols = 6

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, height))

    axs = axs.flatten()

    for ax, feature_index in zip(axs, selected_feature_indices):
        plot_ale(ale_exp_1, features=[feature_index], ax=ax, line_kw={'label': 'Random Forest'})
        plot_ale(ale_exp_2, features=[feature_index], ax=ax, line_kw={'label': 'Gradient Boosting'})

        # set another y-axis scale for hour feature
        if feature_index in large_scale_index:
            ax.set_ylabel('ALE (large scale!)')
            ax.set_ylim([-430,900])
        else:
            ax.set_ylim([-200,200])

    # Disable remaining axes
    for ax in axs[len(selected_features):]:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=spacing)
    plt.suptitle(f'ALE plot for {title}')
    plt.savefig(f'../reports/figures/model_interpretation/ale_{title}', dpi=300)
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
    new_image.save(filename_final, dpi=(300, 300), quality=100, subsampling=0)
    display(new_image)



def plot_boroughs_bar(shap_values, x_data, boroughs, global_min, global_max, filename_bar_plots, filename_final_boroughs):

    # Get the indices in the shap_values where each borough is true
    borough_indices = {borough: np.where(x_data['start_borough_' + borough] == 1)[0] for borough in boroughs}

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
    new_image.save(filename_final_boroughs, dpi=(300, 300), quality=100, subsampling=0)
    display(new_image)


