from local_pkgs.proj_pkg.graph_settings import default_color_palette, default_axis_color
from local_pkgs.proj_pkg.utils import get_text_color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import dcor

def histogram_plot(target, title, target_name=None, units=None, bins=30, savedir=None, takelog=False, color=None):
    plt.figure(figsize=(10, 8))

    if color is None:
        color = default_color_palette[0]

    if takelog:
        # target = target[target > 0] # Filter out non-positive values for log scale
        plt.xscale('log')  # Apply log scale to the x-axis
    
    # Plot the histogram with KDE on original or log-scaled x-axis
    plt.grid(alpha=0.5)


    sns.histplot(target, kde=True, color=color, edgecolor='black', bins=bins)
    
    # Set title and labels
    plt.title(title, pad=20)
    target_name = target_name if target_name else 'Values'
    units = f" ({units})" if units else ''
    plt.xlabel(f"{target_name}{units}",labelpad=10)
    plt.ylabel("Count", labelpad=10)

    plt.tick_params(axis='both', which='major', length=12, width=1.2)
    plt.tick_params(axis='both', which='minor', length=8, width=1)



    # Format x-axis ticks as powers of 10 if takelog is True
    # if takelog:
    #     plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$'))
    

    plt.tight_layout()
    # Save the plot if save directory is provided
    if savedir:
        os.makedirs(os.path.dirname(savedir), exist_ok=True)
        plt.savefig(savedir)
    
    plt.show()


def plot_pearson_correlation_heatmap(
    df, 
    columns, 
    title,
    shape="square", 
    colormap="coolwarm", 
    cbar_range=(-1, 1), 
    cbar_shrink=0.8, 
    savefig=None,
    show_diagonal=True,
):
    """
    Generate a Pearson correlation heatmap with an option for triangular masking and custom colormap.
    Includes the number of data points as annotations, with the diagonal showing only the count.
    Optionally saves the figure to a file. The color bar can be adjusted for position and shrinkage.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    columns : list of str
        List of columns to calculate the correlations for.
    shape : str, optional
        "square" for full matrix, "upper" for upper triangle, "lower" for lower triangle.
    colormap : str, optional
        Colormap for the heatmap.
    cbar_range : tuple of float, optional
        Range for the color bar (default: (-1, 1)).
    cbar_shrink : float, optional
        Shrinkage of the color bar (default: 0.8).
    savefig : str, optional
        Path to save the figure. If None, the figure is not saved.
    show_diagonal : bool, optional
        Whether to show the diagonal values (default: True).

    Raises
    ------
    ValueError
        If cbar_range is outside of [-1, 1].
    """
    # if cbar_range outside of [-1, 1], raise an error
    if cbar_range[0] < -1 or cbar_range[1] > 1:
        raise ValueError("cbar_range must be within the range of -1 and 1.")

    # Ensure columns are numeric
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    
    # Initialize correlation and count matrices
    correlation_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    count_matrix = pd.DataFrame(index=columns, columns=columns, dtype=int)
    
    # Prepare annotations
    annotations = pd.DataFrame(index=columns, columns=columns, dtype=object)
    
    # Compute Pearson correlation and number of data points for each pair
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:  # Self-correlation
                correlation_matrix.loc[col1, col2] = 1  # Set diagonal correlation to 1
                count_matrix.loc[col1, col2] = numeric_df[col1].notna().sum()
                annotations.loc[col1, col2] = f"{int(count_matrix.loc[col1, col2])}"  # Diagonal: count only
            else:
                valid_data = numeric_df[[col1, col2]].dropna()
                if not valid_data.empty:
                    correlation_matrix.loc[col1, col2] = valid_data[col1].corr(valid_data[col2])
                    count_matrix.loc[col1, col2] = len(valid_data)
                    annotations.loc[col1, col2] = f"{int(count_matrix.loc[col1, col2])}"  # Off-diagonal: counts
                else:
                    correlation_matrix.loc[col1, col2] = np.nan
                    count_matrix.loc[col1, col2] = 0
                    annotations.loc[col1, col2] = ""  # No annotation for missing values

    # Apply triangle masking if specified
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    if shape == "upper":
        mask[np.tril_indices_from(mask)] = True  # Mask lower triangle
    elif shape == "lower":
        mask[np.triu_indices_from(mask)] = True  # Mask upper triangle
    if show_diagonal:
        mask[np.eye(mask.shape[0], dtype=bool)] = False  # Unmask diagonal

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=False,
        fmt="s",
        mask=mask,
        cmap=colormap,
        vmin=cbar_range[0],
        vmax=cbar_range[1],
        cbar_kws={"shrink": cbar_shrink},
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"color": "black", "fontsize": 16}
    )

    # Add custom annotations with dynamic text color
    for i in range(len(columns)):
        for j in range(len(columns)):
            if mask[i, j]:  # Skip masked cells
                continue
            value = correlation_matrix.iloc[i, j]
            if pd.notna(value):  # Check if the value is not NaN
                text_color = get_text_color(value, plt.cm.get_cmap(colormap), cbar_range[0], cbar_range[1])
                heatmap.text(j + 0.5, i + 0.5, annotations.iloc[i, j],
                             ha="center", va="center", color=text_color, fontsize=16, zorder=3)

    heatmap.set_xlim(0, len(columns) + 0.02)
    heatmap.set_ylim(len(columns) + 0.02, 0)

    plt.title(title, fontsize=24, pad=20)
    plt.tight_layout(rect=[0, 0, 1.02, 1.02])
    
    # Save the figure if savefig is provided
    if savefig:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, bbox_inches='tight')
        print(f"Figure saved to: {savefig}")
    
    plt.show()

def plot_pearson_correlation_full_heatmap(
    df,
    columns,
    title,
    colormap="coolwarm",
    cbar_range=(-1, 1),
    cbar_shrink=1,
    savefig=None,
):
    """
    Plot a full heatmap of Pearson correlation coefficients, whereby the bottom triangle shows the correlation values, the diagonal shows the number of datapoints for each column, and top triangle shows the number of intersecting datapoints between a pair of columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be correlated.

    columns : list of str
        The list of column names to be correlated.

    title : str
        The title of the heatmap plot.

    colormap : str, optional
        The colormap to use for the heatmap.

    cbar_range : tuple of float, optional
        The range of values for the colorbar.

    cbar_shrink : float, optional
        The shrinkage factor for the colorbar.

    savefig : str, optional
        The file path to save the figure to.

    Raises
    ------
    ValueError
        If cbar_range is outside of [-1, 1].

    """
    # if cbar_range outside of [-1, 1], raise an error
    if cbar_range[0] < -1 or cbar_range[1] > 1:
        raise ValueError("cbar_range must be within the range of -1 and 1.")

    # Ensure columns are numeric
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    
    # Initialize correlation and count matrices
    correlation_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    count_matrix = pd.DataFrame(index=columns, columns=columns, dtype=int)
    
    # Compute Pearson correlation and number of data points for each pair
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:  # Self-correlation
                correlation_matrix.loc[col1, col2] = 1
                count_matrix.loc[col1, col2] = numeric_df[col1].notna().sum()
            else:
                valid_data = numeric_df[[col1, col2]].dropna()
                if not valid_data.empty:
                    correlation_matrix.loc[col1, col2] = valid_data[col1].corr(valid_data[col2])
                    count_matrix.loc[col1, col2] = len(valid_data)
                else:
                    correlation_matrix.loc[col1, col2] = np.nan
                    count_matrix.loc[col1, col2] = 0

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=False,  # Don't add annotations here; we'll add them manually
        fmt="s",
        cmap=colormap,
        vmin=cbar_range[0],
        vmax=cbar_range[1],
        cbar_kws={"shrink": cbar_shrink},
        square=True,
        linewidths=0.5,
        linecolor='lightgray',
        zorder=1
    )

    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=16)  # Adjust fontsize as needed
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=16)  # Adjust fontsize as needed

    colorbar = heatmap.collections[0].colorbar  # Get the colorbar
    colorbar.ax.tick_params(labelsize=16)  # Adjust fontsize as needed

    # Add custom annotations
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i > j:  # Bottom triangle
                text = f"{correlation_matrix.iloc[i, j]:.2f}" if not pd.isna(correlation_matrix.iloc[i, j]) else ""
                value = correlation_matrix.iloc[i, j]
                text_color = get_text_color(value, plt.cm.get_cmap(colormap), cbar_range[0], cbar_range[1])
                heatmap.text(j + 0.5, i + 0.5, text, ha="center", va="center", color=text_color, fontsize=16, zorder=3)
            else:  # Diagonal and top triangle
                text = f"{int(count_matrix.iloc[i, j])}"
                heatmap.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="gray", fontsize=14, fontweight='light', zorder=3)

    # Overlay white mask on the diagonal and top triangle
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i <= j:  # Diagonal and top triangle
                heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor="white",edgecolor='lightgray', lw=0.5, alpha=1, zorder=2))
    heatmap.set_xlim(0, len(columns) + 0.02)
    heatmap.set_ylim(len(columns) + 0.02, 0)

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout(rect=[0, 0, 1.02, 1.02])

    # Save the figure if savefig is provided
    if savefig:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, bbox_inches="tight")
        print(f"Figure saved to: {savefig}")
    
    plt.show()

def plot_distance_correlation_full_heatmap(
    df,
    columns,
    title,
    colormap="coolwarm",
    cbar_range=(0, 1),
    cbar_shrink=1,
    savefig=None,
):
    """
    Plot a full heatmap of distance correlation coefficients, whereby the bottom triangle shows the distance correlation values,
    the diagonal shows the number of datapoints for each column, and the top triangle shows the number of intersecting datapoints
    between a pair of columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be correlated.
    columns : list of str
        The list of column names to be correlated.
    title : str
        The title of the heatmap plot.
    colormap : str, optional
        The colormap to use for the heatmap.
    cbar_range : tuple of float, optional
        The range of values for the colorbar.
    cbar_shrink : float, optional
        The shrinkage factor for the colorbar.
    savefig : str, optional
        The file path to save the figure to.

    Raises
    ------
    ValueError
        If cbar_range is outside of [0, 1].
    """
    if cbar_range[0] < 0 or cbar_range[1] > 1:
        raise ValueError("cbar_range must be within the range of 0 and 1.")

    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    correlation_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    count_matrix = pd.DataFrame(index=columns, columns=columns, dtype=int)

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                correlation_matrix.loc[col1, col2] = 1
                count_matrix.loc[col1, col2] = numeric_df[col1].notna().sum()
            else:
                valid_data = numeric_df[[col1, col2]].dropna()
                if not valid_data.empty:
                    correlation_matrix.loc[col1, col2] = dcor.distance_correlation(valid_data[col1], valid_data[col2])
                    count_matrix.loc[col1, col2] = len(valid_data)
                else:
                    correlation_matrix.loc[col1, col2] = np.nan
                    count_matrix.loc[col1, col2] = 0

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=False,
        fmt="s",
        cmap=colormap,
        vmin=cbar_range[0],
        vmax=cbar_range[1],
        cbar_kws={"shrink": cbar_shrink},
        square=True,
        linewidths=0.5,
        linecolor='lightgray',
        zorder=1
    )

    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=16)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=16)
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=16)

    for i in range(len(columns)):
        for j in range(len(columns)):
            if i > j:
                text = f"{correlation_matrix.iloc[i, j]:.2f}" if not pd.isna(correlation_matrix.iloc[i, j]) else ""
                value = correlation_matrix.iloc[i, j]
                text_color = get_text_color(value, plt.cm.get_cmap(colormap), cbar_range[0], cbar_range[1])
                heatmap.text(j + 0.5, i + 0.5, text, ha="center", va="center", color=text_color, fontsize=16, zorder=3)
            else:
                text = f"{int(count_matrix.iloc[i, j])}"
                heatmap.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="gray", fontsize=14, fontweight='light', zorder=3)

    for i in range(len(columns)):
        for j in range(len(columns)):
            if i <= j:
                heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor="white", edgecolor='lightgray', lw=0.5, alpha=1, zorder=2))
    heatmap.set_xlim(0, len(columns) + 0.02)
    heatmap.set_ylim(len(columns) + 0.02, 0)

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout(rect=[0, 0, 1.02, 1.02])

    if savefig:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, bbox_inches="tight")
        print(f"Figure saved to: {savefig}")

    plt.show()