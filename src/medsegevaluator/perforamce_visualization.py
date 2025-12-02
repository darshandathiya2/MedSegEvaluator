import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from skimage import measure

def visualize_image_contour(image, gt_image, inf_image, slice_index=76):
    """
    Display a slice with GT (yellow) and prediction (green) contours overlaid on the original image.

    Args:
        image (np.ndarray): 3D input image (H, W, D)
        gt_image (np.ndarray): 3D ground truth mask (H, W, D)
        inf_image (np.ndarray): 3D predicted mask (H, W, D)
        slice_index (int): slice index to display
    """
    # Extract single slices
    img = image[:, :, slice_index]
    gt = gt_image[:, :, slice_index]
    pred = inf_image[:, :, slice_index]

    # Normalize image for display
    img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-8)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_norm, cmap='gray')

    # Find contours for GT and prediction
    gt_contours = measure.find_contours(gt, level=0.5)
    pred_contours = measure.find_contours(pred, level=0.5)

    # Plot contours
    for contour in gt_contours:
        plt.plot(contour[:, 1], contour[:, 0], color='yellow', linewidth=2, label='Ground Truth')

    for contour in pred_contours:
        plt.plot(contour[:, 1], contour[:, 0], color='lime', linewidth=2, label='Prediction')

    plt.title(f"Slice {slice_index} - T1C Image")
    plt.axis('off')

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_histogram_comparison(
    data1,
    data2,
    label1="Set 1",
    label2="Set 2",
    xlabel="Value",
    ylabel="Count",
    title="Distribution Comparison",
    caption=None,
    bins=20,
    colors=("skyblue", "salmon"),
    save_path=None
                    ):
    """
    Plot overlapping histogram comparison between two numeric datasets.

    Parameters
    ----------
    data1, data2 : list, np.ndarray, or pd.Series
        Numeric data to plot.
    label1, label2 : str
        Legends for datasets.
    xlabel, ylabel, title : str
        Axis and title labels.
    caption : str or None
        Optional caption displayed below the plot.
    bins : int
        Number of bins.
    colors : tuple
        Colors for the histograms.
    save_path : str or None
        Path to save figure (optional).
    """

    # Convert DataFrame columns if passed as pd.Series
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Handle dynamic bin ranges
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    bins = np.linspace(min_val, max_val, bins)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.hist(data1, bins=bins, alpha=0.4, label=label1, color=colors[0], edgecolor="black")
    plt.hist(data2, bins=bins, alpha=0.4, label=label2, color=colors[1], edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Optional caption
    if caption:
        plt.figtext(0.5, -0.05, caption, wrap=True, ha='center', fontsize=9, style='italic')

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Plot generated for: {title}")


def plot_boxplot_by_islands(
    data_or_islands,
    dice_values=None,
    xlabel='Ground-truth island count category',
    ylabel='Dice coefficient',
    title='Dice distribution by island count grouping',
    caption=None
      ):
    """
    Plots a box plot comparing Dice scores for single vs multiple islands.

    Parameters:
        data_or_islands: 
            - pd.DataFrame containing columns ['GT_Islands', 'Dice'], or
            - array-like/list of island counts, or
            - a Series for island counts.
        dice_values: 
            - Optional, array-like/list/Series for Dice coefficients (if data_or_islands is not a DataFrame).
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Title of the plot.
        caption (str, optional): Optional caption below the plot.
    """

    # --- Case 1: DataFrame input ---
    if isinstance(data_or_islands, pd.DataFrame):
        df = data_or_islands.copy()
        if 'GT_Islands' not in df.columns or 'Dice' not in df.columns:
            raise ValueError("DataFrame must contain 'GT_Islands' and 'Dice' columns.")
    # --- Case 2: Arrays/lists input ---
    else:
        if dice_values is None:
            raise ValueError("If not passing a DataFrame, please provide both island counts and dice values.")
        df = pd.DataFrame({
            'GT_Islands': np.array(data_or_islands),
            'Dice': np.array(dice_values)
        })

    # --- Create grouping column (1 island vs >1 islands) ---
    df['GT_is_single_island'] = df['GT_Islands'].apply(lambda x: 1 if x == 1 else 0)

    # --- Initialize plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df.dropna(subset=['GT_is_single_island', 'Dice']),
        x='GT_is_single_island',
        y='Dice',
        ax=ax
    )

    # --- Customize appearance ---
    ax.set_xticklabels(['1 island', '>1 islands'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    # --- Optional caption ---
    if caption:
        plt.figtext(0.5, -0.05, caption, ha='center', fontsize=9, color='gray')

    plt.show()
