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

def consolidate_scores(
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