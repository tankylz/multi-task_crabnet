import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import os
from local_pkgs.proj_pkg.utils import load_dataframe, adjust_hsv, equalsIgnoreCaseAndSpace
from local_pkgs.proj_pkg.graph_settings import default_color_palette, default_axis_color

def valid_metric(metric):
    '''
    Helper Function to ensure that the metric parsed into the barplot is valid
    
    Parameters
    ----------
    metric : str
        A metric to be used for visualization. It can be 'RMSE (meV / atom)' or 'R-Squared'.
    
    '''
    metric_ls = ['RMSE (meV / atom)', 'R-Squared']
    
    for m in metric_ls:
        if equalsIgnoreCaseAndSpace(metric, m):
            return
    
    raise ValueError("Invalid metric. Please choose from 'RMSE (meV / atom)' or 'R-Squared'.")

# parity plot

# epoch vs loss


# interactive graph for table

# # Load the dataset
# file_path = 'TE Files/TE_with_composition_2024-10-03-V1.xlsx' 
# df = pd.read_excel(file_path)

# # Separate numeric and categorical columns
# numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
# categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
# categorical_columns.insert(0, None)  # Allow for no grouping

# # Create dropdown menus for selecting columns and grouping
# x_dropdown = widgets.Dropdown(options=numeric_columns, description='X-axis:')
# y_dropdown = widgets.Dropdown(options=numeric_columns, description='Y-axis:')
# group_dropdown = widgets.Dropdown(options=categorical_columns, description='Group By:')
# fill_na_dropdown = widgets.Dropdown(options=['None', 'Zero', 'Mean'], description='Fill NaN:')

# Define the update function for the plot
# def update_plot(x_column, y_column, group_by, fill_na):
#     # Handle missing data
#     df_cleaned = df.copy()
#     if fill_na == 'Zero':
#         df_cleaned = df_cleaned.fillna(0)
#     elif fill_na == 'Mean':
#         df_cleaned = df_cleaned.fillna(df_cleaned.mean())
    
#     # Generate the plot
#     if x_column and y_column:
#         if group_by:
#             fig = px.bar(df_cleaned, x=group_by, y=y_column, color=x_column, 
#                          title=f'{y_column} vs {group_by} grouped by {x_column}')
#         else:
#             fig = px.scatter(df_cleaned, x=x_column, y=y_column, 
#                              title=f'{x_column} vs {y_column}')
#         fig.show()

# Link the dropdowns to the update function
# widgets.interactive(update_plot, x_column=x_dropdown, y_column=y_dropdown, group_by=group_dropdown, fill_na=fill_na_dropdown)

# Display the widgets
# display(x_dropdown, y_dropdown, group_dropdown, fill_na_dropdown)

# Feature Importance Plot


def plot_top_feature_importances(model, feature_names, top_n_features=10, show_sum=True):
    """
    Plot the top N feature importances from a given model.
    
    Parameters:
        model : object
            A trained model with a 'feature_importances_' attribute.
        feature_names : list or array-like
            List of feature names corresponding to the columns in the dataset.
        top_n_features : int
            The number of top features to plot.
        show_sum : bool
            Whether to display the total feature importance of the top n features in the title.
    """
    # Check if the model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        raise ValueError(f"The provided model, {type(model)} does not have a 'feature_importances_' attribute.")
    
    feature_importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    
    # Get the top N features
    top_indices = indices[:top_n_features]
    top_feature_importances = feature_importances[top_indices]
    top_feature_names = np.array(feature_names)[top_indices]
    
    # Create horizontal bar plot using seaborn's deep color palette
    sns.set_theme(style="whitegrid", font='monospace')
    plt.figure(figsize=(10, 8))
    
    plt.barh(range(top_n_features), top_feature_importances, align="center", color=default_color_palette[0], alpha=0.9)
    plt.yticks(range(top_n_features), top_feature_names)
    plt.gca().invert_yaxis()  # Highest importance at the top
    if show_sum:
        sum_importances = np.sum(top_feature_importances)
        plt.title(f"Top {top_n_features} Feature Importances, Total: {sum_importances:.2f}")

    else:
        plt.title(f"Top {top_n_features} Feature Importances")
    # plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()

def parity_plot_simple(y_true, y_pred, title, target_name=None, units=None, r_squared=True, mae=True, rmse=True, decimal_points=2, savedir=None, tol = 0.02, zoom_min=None, zoom_max=None, take_log=False):
    if take_log:
        if (zoom_min and zoom_min <= 0) or (zoom_max and zoom_max <= 0):
            raise ValueError("The minimum value for the log scale must be greater than 0.")
        

    # Create the plot
    plt.figure(figsize=(10, 8))

    # sns.set_theme(style='whitegrid', font='monospace')
    # Create range of values for the plot
    scaling = max(y_pred.max(), y_true.max()) - min(y_pred.min(), y_true.min())
    min_bound = min(y_pred.min() - tol * scaling, y_true.min() - tol * scaling)
    max_bound = max(y_pred.max() + tol * scaling, y_true.max() + tol * scaling)

    if take_log:
        if min_bound <=0 or max_bound <=0:
            raise ValueError("The minimum value for the log scale must be greater than 0. Try setting a different tolerance value.")
        

    sns.lineplot(x=[min_bound, max_bound], y=[min_bound, max_bound], linestyle='--', color='black', linewidth=1.5, alpha=0.8, zorder=2)
    # sns.scatterplot(x=y_true, y=y_pred, facecolors="none", edgecolor=sns.color_palette('deep')[0], s=5, alpha=0.9, zorder=1)
    sns.scatterplot(x=y_true, y=y_pred, facecolors=sns.color_palette('deep')[0], edgecolor='black', s=20, alpha=1.0, zorder=1)

    if target_name == None:
        target_name = 'Values'
    if units == None:
        units = ''
        bracketed_units = ''
    else:
        bracketed_units = f" ({units})"

    plt.xlabel(f"True {target_name}{bracketed_units}", fontsize=14, labelpad=10)
    plt.ylabel(f"Predicted {target_name}{bracketed_units}", fontsize=14, labelpad=10)
    plt.title(title, fontsize=18, fontweight='bold', pad=20)

    r2_text=''
    mae_text=''
    rmse_text=''
    ax = plt.gca()

    # Calculate metrics
    r2 = None
    mae_score = None
    rmse_score = None

    if r_squared:
        r2 = r2_score(y_true, y_pred)
        r2_text = f"$R^2$: {r2:.{decimal_points}f}"
    if mae:
        mae_score = mean_absolute_error(y_true, y_pred)
        mae_text = f"MAE: {mae_score:.{decimal_points}f} {units}"
    if rmse:
        rmse_score = root_mean_squared_error(y_true, y_pred)
        rmse_text = f"RMSE: {rmse_score:.{decimal_points}f} {units}"
    
    if mae or rmse or r_squared:
        overall_text = f"{r2_text}\n{mae_text}\n{rmse_text}"
        plt.text(0.05, 0.95, overall_text, fontsize=10, transform=ax.transAxes,verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.03))
    
    if take_log:
        plt.xscale('log')
        plt.yscale('log')

    if zoom_min or zoom_max:
        if zoom_min == None:
            zoom_min = min_bound
        if zoom_max == None:
            zoom_max = max_bound
        plt.xlim(zoom_min, zoom_max)
        plt.ylim(zoom_min, zoom_max)
    else:
        plt.xlim(min_bound, max_bound)
        plt.ylim(min_bound, max_bound)

    # ax.set_aspect('equal', adjustable='box')

    if savedir:
        # make the directory if it doesn't exist
        if not os.path.exists(os.path.dirname(savedir)):
            os.makedirs(os.path.dirname(savedir))
        plt.savefig(savedir)
    
    plt.show()    
    return r2, mae_score, rmse_score

def parity_plot(
    y_true,
    y_pred,
    title,
    target_name=None,
    units=None,
    r_squared=True,
    mae=True,
    rmse=True,
    decimal_points=2,
    savedir=None,
    tol=0.02,
    zoom_min=None,
    zoom_max=None,
    take_log=False,
    fold_column=None
):
    if take_log:
        if (zoom_min and zoom_min <= 0) or (zoom_max and zoom_max <= 0):
            raise ValueError("The minimum value for the log scale must be greater than 0.")

    # Create the plot
    plt.figure(figsize=(10, 8))
    scaling = max(y_pred.max(), y_true.max()) - min(y_pred.min(), y_true.min())
    min_bound = min(y_pred.min() - tol * scaling, y_true.min() - tol * scaling)
    max_bound = max(y_pred.max() + tol * scaling, y_true.max() + tol * scaling)
    plt.grid(True) 
    plt.gca().set_axisbelow(True)  # Ensure the axis gridlines are below the plots

    if take_log:
        if min_bound <= 0 or max_bound <= 0:
            raise ValueError("The minimum value for the log scale must be greater than 0. Try setting a different tolerance value.")

    sns.lineplot(x=[min_bound, max_bound], y=[min_bound, max_bound], linestyle='--', color='black', linewidth=1.5, alpha=0.8, zorder=2)
    sns.scatterplot(x=y_true, y=y_pred, facecolors=sns.color_palette('deep')[0], edgecolor='black', s=20, alpha=1.0, zorder=1)

    if target_name is None:
        target_name = 'Values'
    if units is None:
        units = ''
        bracketed_units = ''
    else:
        bracketed_units = f" ({units})"

    plt.xlabel(f"True {target_name}{bracketed_units}", fontsize=18, labelpad=10)
    plt.ylabel(f"Predicted {target_name}{bracketed_units}", fontsize=18, labelpad=10)
    plt.title(title, fontsize=18, fontweight='bold', pad=10)

    r2_text, mae_text, rmse_text = '', '', ''
    ax = plt.gca()

    # Initialize variables
    r2, mae_score, rmse_score = None, None, None
    r2_std_error, mae_std_error, rmse_std_error = None, None, None

    if fold_column is not None:
        # Group by fold column and compute metrics
        unique_folds = np.unique(fold_column)
        r2_scores, mae_scores, rmse_scores = [], [], []

        for fold in unique_folds:
            indices = fold_column == fold
            fold_y_true = y_true[indices]
            fold_y_pred = y_pred[indices]

            if r_squared:
                r2_scores.append(r2_score(fold_y_true, fold_y_pred))
            if mae:
                mae_scores.append(mean_absolute_error(fold_y_true, fold_y_pred))
            if rmse:
                rmse_scores.append(root_mean_squared_error(fold_y_true, fold_y_pred))

        # Calculate means and standard errors
        if r_squared:
            r2 = np.mean(r2_scores)
            r2_std_error = np.std(r2_scores) / np.sqrt(len(unique_folds))
        if mae:
            mae_score = np.mean(mae_scores)
            mae_std_error = np.std(mae_scores) / np.sqrt(len(unique_folds))
        if rmse:
            rmse_score = np.mean(rmse_scores)
            rmse_std_error = np.std(rmse_scores) / np.sqrt(len(unique_folds))

    else:
        # Calculate metrics on the entire dataset
        if r_squared:
            r2 = r2_score(y_true, y_pred)
        if mae:
            mae_score = mean_absolute_error(y_true, y_pred)
        if rmse:
            rmse_score = root_mean_squared_error(y_true, y_pred)

    # Format the metrics text
    if r_squared:
        r2_text = f"$R^2$: {r2:.{decimal_points}f} ± {r2_std_error:.{decimal_points}f}" if r2_std_error else f"$R^2$: {r2:.{decimal_points}f}"
    if mae:
        mae_text = f"MAE: {mae_score:.{decimal_points}f} ± {mae_std_error:.{decimal_points}f} {units}" if mae_std_error else f"MAE: {mae_score:.{decimal_points}f} {units}"
    if rmse:
        rmse_text = f"RMSE: {rmse_score:.{decimal_points}f} ± {rmse_std_error:.{decimal_points}f} {units}" if rmse_std_error else f"RMSE: {rmse_score:.{decimal_points}f} {units}"

    if mae or rmse or r_squared:
        overall_text = f"{r2_text}\n{mae_text}\n{rmse_text}"
        plt.text(0.05, 0.95, overall_text, fontsize=18, transform=ax.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.03))

    if take_log:
        plt.xscale('log')
        plt.yscale('log')

    if zoom_min or zoom_max:
        plt.xlim(zoom_min if zoom_min else min_bound, zoom_max if zoom_max else max_bound)
        plt.ylim(zoom_min if zoom_min else min_bound, zoom_max if zoom_max else max_bound)
    else:
        plt.xlim(min_bound, max_bound)
        plt.ylim(min_bound, max_bound)

    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to equal

    if savedir:
        if not os.path.exists(os.path.dirname(savedir)):
            os.makedirs(os.path.dirname(savedir))
        plt.savefig(savedir)

    plt.show()

    return {
        "r2_mean": r2,
        "r2_se": r2_std_error,
        "mae_mean": mae_score,
        "mae_se": mae_std_error,
        "rmse_mean": rmse_score,
        "rmse_se": rmse_std_error
    }

def scores_ls(
    y_true,
    y_pred,
    fold_column=None,
    r_squared=True,
    mae=True,
    rmse=True
):
    """
    Computes and returns lists of R^2, MAE, and RMSE scores for each fold.
    
    Parameters:
    - y_true: Array of true target values.
    - y_pred: Array of predicted values.
    - fold_column: Array indicating the fold assignment for each data point (optional).
    - r_squared: Boolean to compute R^2 scores (default: True).
    - mae: Boolean to compute MAE scores (default: True).
    - rmse: Boolean to compute RMSE scores (default: True).

    Returns:
    - A dictionary with keys 'r2_scores', 'mae_scores', 'rmse_scores' containing respective lists of scores.
    """
    # Initialize lists for metrics
    r2_scores, mae_scores, rmse_scores = [], [], []

    if fold_column is not None:
        # Group by fold column and compute metrics
        unique_folds = np.unique(fold_column)

        for fold in unique_folds:
            indices = fold_column == fold
            fold_y_true = y_true[indices]
            fold_y_pred = y_pred[indices]

            if r_squared:
                r2_scores.append(r2_score(fold_y_true, fold_y_pred))
            if mae:
                mae_scores.append(mean_absolute_error(fold_y_true, fold_y_pred))
            if rmse:
                rmse_scores.append(root_mean_squared_error(fold_y_true, fold_y_pred))
    else:
        # Compute metrics on the entire dataset
        if r_squared:
            r2_scores.append(r2_score(y_true, y_pred))
        if mae:
            mae_scores.append(mean_absolute_error(y_true, y_pred))
        if rmse:
            rmse_scores.append(root_mean_squared_error(y_true, y_pred))

    # Return lists of scores
    return {
        "r2_scores": r2_scores,
        "mae_scores": mae_scores,
        "rmse_scores": rmse_scores
    }





# def plot_h_bar(input_data, material_class, metric, bar_height=0.4, internal_gap=0.1, group_gap=0.4, figsize=(30, 16), config=None):
#     '''
#     Generates a horizontal bar chart to compare a metric (with mean and std dev) across different ML models for different materials.
    
#     Parameters
#     ----------
#     input_data : str or pd.DataFrame
#         File path to the CSV file or a DataFrame.
        
#     material_class : str
#         The class of materials, which will be used as the title. Examples: 'Binary Alloys', 'Perovskites'.
                
#     metric : str
#         The metric used to generate the horizontal bar chart. It can be 'RMSE (meV / atom)' or 'R-Squared'.
        
#     bar_height : float, optional
#         The height of each bar. The default is 0.4.
        
#     internal_gap : float, optional
#         The gap between bars. The default is 0.1.
        
#     group_gap : float, optional
#         An additional gap between bars of different materials. This gap will help to create visual separation between different materials. The default is 0.4.
        
#     figsize : tuple, optional
#         The size of the figure. The default is (30, 16).
        
#     config : dict, optional
#         A dictionary containing additional configuration options pertaining to the order of the materials and the colors of the ML models. If None, the results of all ML models in the sheet will be shown, the materials and models will be sorted in ascending order, and default colors will be used. 
#         The default is None.
        
#         Example:
#         {
#             'Material': {
#                 'order': ['BinaryAlloys-MoZr', 'BinaryAlloys-MoNb']
#             },
#             'Type of ML Model': {
#                 'order': ['Linear Ridge', 'KRR RBF'],
#                 'colors': {
#                     'Linear Ridge': 'blue',
#                     'KRR RBF': 'red'
#                 }
#             }
#         }
#     '''
        
#     # Check if the metric is valid and set the color
#     valid_metric(metric)    
    
#     df = load_dataframe(input_data)    
    
#     ml_model_count = len(df['Type of ML Model'].unique())  # Count unique ML models

#     # Handle custom order if provided
#     if config:
#         if 'Material' in config and 'order' in config['Material']:
#             df['Material'] = pd.Categorical(df['Material'], categories=config['Material']['order'], ordered=True)
#         if 'Type of ML Model' in config and 'order' in config['Type of ML Model']:
#             df['Type of ML Model'] = pd.Categorical(df['Type of ML Model'], categories=config['Type of ML Model']['order'], ordered=True)
#             ml_model_count = len(config['Type of ML Model']['order'])  # Count unique ML models
            
#     df_sorted = df.sort_values(by=['Material', 'Type of ML Model'])
    
#     fig, ax1 = plt.subplots(figsize=figsize)
    
#     valid_combinations = df_sorted.groupby(['Material', 'Type of ML Model'], observed=True).size().index.tolist()

#     y_data = [] # appends the center positions of the bars
#     material_labels = [] # example ['Binary Alloys', 'Perovskites']
#     ml_model_labels = [] # example ['Linear Ridge', 'KRR RBF']
#     y_pos = 0
#     material_system_positions = {} # contains {material_system: middle_position}

#     current_material_system = None
    

#     model_iter = 0
    
#     for ms, ml in valid_combinations:
#         if ms != current_material_system:
#             if current_material_system is not None:
#                 y_pos += group_gap # add a gap between Materials
            
#             init_y_pos = y_pos 
#             model_iter = 0
            
#             current_material_system = ms
        
#         y_pos += (bar_height + internal_gap)
#         ml_model_labels.append(ml)
#         y_data.append(y_pos)
        
#         # if model_iter == 0:
#         #     # add a light horizontal line below to separate the materials
#         #     ax1.axhline(y= y_pos - (bar_height + internal_gap) / 2, color=default_axis_color, linestyle='-', lw=2, alpha = 0.04)
            
#         if model_iter == ml_model_count - 1:
#             middle_position = (init_y_pos + y_pos + (bar_height + internal_gap)) / 2
#             material_system_positions[ms] = middle_position
#             material_labels.append(ms.split('-')[1])

#             # add a light horizontal line above to separate the materials
#             # ax1.axhline(y= y_pos + (bar_height + internal_gap) / 2, color=default_axis_color, linestyle='-', lw=2, alpha = 0.04)
        
#         model_iter += 1
    
#     metric_mean = []
#     metric_std = []
#     bar_colors = []
#     legend_handles = []
#     unique_ml_models = set()
    
#     count = 0 # used to vary the default color palette
#     for ms, ml in valid_combinations:
#         filtered = df_sorted[(df_sorted['Material'] == ms) & (df_sorted['Type of ML Model'] == ml)]
#         if not filtered.empty:
#             metric_mean.append(filtered[f'Mean {metric}'].values[0])
#             metric_std.append(filtered[f'Std Dev {metric}'].values[0])
#             if config and 'Type of ML Model' in config and 'colors' in config['Type of ML Model']:
#                 color = config['Type of ML Model']['colors'].get(ml, default_axis_color)
#             else:
#                 color = default_color_palette[count]
#                 count += 1
#             bar_colors.append(color)
            
#             if ml not in unique_ml_models:
#                 unique_ml_models.add(ml)
#                 legend_handles.append(Patch(color=color, label=ml))
    
#     legend_handles = legend_handles[::-1] # reverse the order of the legend handles
#     metric_mean = np.array(metric_mean)
#     metric_std = np.array(metric_std)
#     y_data = np.array(y_data)
    
#     # Plot the bar chart
#     metric_bar = ax1.barh(y_data, metric_mean, bar_height, xerr=metric_std, color=bar_colors, error_kw=dict(lw=1.5, capsize=4, capthick=1.5), alpha=0.99)
    
#     # If metric is R-Squared, draw a dotted line at R-squared = 1
#     if metric == 'R-Squared':
#         ax1.axvline(x=1, color=default_axis_color, linestyle=(0, (10, 10)), lw=1.5, alpha = 0.2) # linestyle is customized to make it sparser
        
#     ax1.set_title(f'{material_class}', fontsize=24)
#     ax1.set_yticks(y_data)
#     ax1.tick_params(axis='y', which='both', length=0) # Removes the ticks
#     ax1.set_yticklabels(['' for _ in ml_model_labels], ha='left') # Removes the default tick labels

#     # Set the x-axis limits for the number of features
#     # x_min = (metric_mean - metric_std).min() - 0.1 * (metric_mean - metric_std).min()
#     x_min = 0 # always start bar plots from 0
#     x_max = (metric_mean + metric_std).max() + 0.1 * (metric_mean + metric_std).max()
#     x_range = x_max - x_min
#     ax1.set_xlim(x_min, x_max) 

#     ax1.set_xlabel(f'{metric}', color=default_axis_color, fontsize=20)
#     ax1.tick_params(axis='x', colors=default_axis_color, labelsize=14)
#     ax1.spines['bottom'].set_color(default_axis_color)
#     ax1.spines['bottom'].set_linewidth(1.6)

#     ax1.grid(axis='x', which='both', linewidth=0.2)
#     ax1.yaxis.grid(False)

#     # Add text after setting axis limits
#     for ms, ypos in material_system_positions.items():
#         # add text indicating the material
#         ax1.text(x_min - 0.1 * x_range, ypos, ms.split('-')[1], ha='right', va='center', rotation=90, fontsize=20, fontweight='bold')

#     for ypos, values, color in zip(y_data, metric_mean, bar_colors):
#         # add text indicating the metric values
#         ax1.text(values + 0.06 * x_range, ypos - 0.5 * bar_height, f'{values}', ha='center', va='center', fontsize=14, color=color)

#     # Add legend
#     ax1.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1, 0), fontsize=10)
#     plt.subplots_adjust(left=0.6, right=0.8, top=0.95, bottom=0.5)

#     # Adjust subplot parameters to control the size of the plot area
#     plt.show()

