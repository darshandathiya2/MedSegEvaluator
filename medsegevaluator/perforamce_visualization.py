import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from skimage import measure

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class PerformanceVisualization:
    r"""
    Visualization utilities for quantitative performance analysis.
    """

    @staticmethod
    def bland_altman_plot_interactive(
        values1,
        values2,
        patient_ids=None,
        title="Bland–Altman Plot",
        xlabel="Mean of Methods",
        ylabel="Difference (GT - Pred)",
        units=None
    ):
        r"""
        Generate an interactive Bland–Altman plot to assess agreement
        between two measurement methods.

        Let two paired measurements be denoted as
        :math:`x_i` (reference / ground truth) and
        :math:`y_i` (predicted values), for :math:`i = 1, \dots, N`.

        The mean and difference for each pair are computed as:

        .. math::

            m_i = \frac{x_i + y_i}{2}

        .. math::

            d_i = x_i - y_i

        The mean difference (bias) is defined as:

        .. math::

            \bar{d} = \frac{1}{N} \sum_{i=1}^{N} d_i

        The standard deviation of the differences is:

        .. math::

            s_d = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (d_i - \bar{d})^2}

        The 95\% limits of agreement (LoA) are computed as:

        .. math::

            \text{LoA}_{\text{upper}} = \bar{d} + 1.96 \cdot s_d

        .. math::

            \text{LoA}_{\text{lower}} = \bar{d} - 1.96 \cdot s_d

        The Bland–Altman plot visualizes :math:`d_i` against :math:`m_i`,
        along with the mean difference and limits of agreement, enabling
        assessment of systematic bias and variability between methods.

        Parameters
        ----------
        values1 : array-like
            First set of measurements (e.g., ground truth).
        values2 : array-like
            Second set of measurements (e.g., model predictions).
        patient_ids : list or None, optional
            Identifiers displayed in hover information.
        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        units : str or None, optional
            Measurement units displayed on axes.

        Returns
        -------
        tuple
            Mean difference, upper limit of agreement, and lower limit of agreement.
        """

        values1 = np.asarray(values1, dtype=float)
        values2 = np.asarray(values2, dtype=float)

        mean_values = (values1 + values2) / 2
        diff = values1 - values2

        mean_diff = np.mean(diff)
        sd_diff = np.std(diff)

        loa_upper = mean_diff + 1.96 * sd_diff
        loa_lower = mean_diff - 1.96 * sd_diff

        if patient_ids is None:
            tooltips = [
                f"Mean: {m:.2f}<br>Diff: {d:.2f}"
                for m, d in zip(mean_values, diff)
            ]
        else:
            tooltips = [
                f"Patient ID: {pid}<br>Mean: {m:.2f}<br>Diff: {d:.2f}"
                for pid, m, d in zip(patient_ids, mean_values, diff)
            ]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=mean_values,
            y=diff,
            mode='markers',
            marker=dict(size=8, color="steelblue"),
            text=tooltips,
            hoverinfo='text',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[min(mean_values), max(mean_values)],
            y=[mean_diff, mean_diff],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Mean Diff = {mean_diff:.2f}"
        ))

        fig.add_trace(go.Scatter(
            x=[min(mean_values), max(mean_values)],
            y=[loa_upper, loa_upper],
            mode="lines",
            line=dict(color="green", width=2, dash="dash"),
            name=f"+1.96 SD = {loa_upper:.2f}"
        ))

        fig.add_trace(go.Scatter(
            x=[min(mean_values), max(mean_values)],
            y=[loa_lower, loa_lower],
            mode="lines",
            line=dict(color="orange", width=2, dash="dash"),
            name=f"-1.96 SD = {loa_lower:.2f}"
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title=xlabel + (f" ({units})" if units else ""),
            yaxis_title=ylabel + (f" ({units})" if units else ""),
            width=800,
            height=650,
            legend=dict(
                orientation="v",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="gray",
                borderwidth=1
            )
        )

        fig.show()

        return mean_diff, loa_upper, loa_lower


'''
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
'''
