import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from skimage import measure
import plotly.graph_objects as go
import plotly.express as px

class PerformanceVisualization:
    r"""
    Visualization utilities for quantitative performance analysis.
    """

    @staticmethod
    def histogram_comparison(
        data1,
        data2,
        label1="Set 1",
        label2="Set 2",
        xlabel="Value",
        ylabel="Count",
        title="Distribution Comparison",
        caption=None,
        bins=20,
        colors=None,
        save_path=None
        ):
        r"""
        Plot an overlapping histogram comparison between two numeric datasets
        using color-blind–friendly colors.
    
        Suitable for comparing Dice, area, volume, and other quantitative
        performance measures.
    
        Parameters
        ----------
        data1, data2 : array-like
            Numeric data.
        label1, label2 : str, optional
            Legends.
        xlabel, ylabel, title : str, optional
            Labels.
        caption : str or None, optional
            Optional caption.
        bins : int, optional
            Number of bins.
        colors : tuple or None, optional
            Custom colors (hex or names). If None, uses color-blind safe palette.
        save_path : str or None, optional
            Path to save figure.
    
        Returns
        -------
        tuple
            (Figure, Axes)
        """
        # Okabe–Ito color-blind safe defaults
        if colors is None:
            colors = ("#0072B2", "#E69F00")  # Blue, Orange
    
        # Convert inputs
        data1 = np.asarray(data1, dtype=float)
        data2 = np.asarray(data2, dtype=float)
    
        # Remove NaNs
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
    
        if data1.size == 0 or data2.size == 0:
            raise ValueError("Input data must not be empty.")
    
        # Dynamic bins
        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
    
        # Plot
        fig, ax = plt.subplots(figsize=(7, 5))
    
        ax.hist(
            data1,
            bins=bin_edges,
            alpha=0.5,
            label=label1,
            color=colors[0],
            edgecolor="black"
        )
    
        ax.hist(
            data2,
            bins=bin_edges,
            alpha=0.5,
            label=label2,
            color=colors[1],
            edgecolor="black"
        )
    
        # Labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    
        fig.tight_layout()
    
        # Caption
        if caption:
            fig.text(
                0.5, -0.08, caption,
                ha="center", fontsize=9, style="italic"
            )
    
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
        return fig, ax


    @staticmethod
    def bland_altman_plot_interactive(
        values1,
        values2,
        patient_ids=None,
        title="Bland–Altman Plot",
        xlabel="Mean of Methods",
        ylabel="Difference (GT - Pred)",
        units=None):
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


    @staticmethod
    def delta_dice_diverging_bar_plot(
        delta_values,
        ids=None,
        title="Diverging Bar Plot of Delta Dice Under Perturbation",
        xlabel="Images",
        ylabel="Δ Dice",
        show=True
        ):
        r"""
        Generate an interactive diverging bar plot to visualize changes
        in Dice score under perturbation.
    
        Positive Δ Dice values (shown in red) indicate performance degradation,
        while negative Δ Dice values (shown in green) indicate performance
        improvement.
    
        This function accepts lists, NumPy arrays, Pandas Series, or
        DataFrame columns as input.
    
        Parameters
        ----------
        delta_values : array-like
            Δ Dice values (list, NumPy array, Pandas Series, or DataFrame column).
        ids : array-like or None, optional
            Image/sample identifiers.
        title : str, optional
            Plot title.
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.
        show : bool, optional
            Whether to display the plot.
    
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure object.
        """

        # Convert delta values to numpy array
        delta = np.asarray(delta_values, dtype=float)
    
        # Validate input
        if delta.ndim != 1:
            raise ValueError("delta_values must be a 1D array-like object.")
    
        n = len(delta)
    
        # Handle IDs
        if ids is None:
            ids = [f"Sample {i+1}" for i in range(n)]
        else:
            ids = np.asarray(ids)
    
            if len(ids) != n:
                raise ValueError("ids must have same length as delta_values.")
    
        # Build DataFrame
        df = pd.DataFrame({
            "index": np.arange(n),
            "delta_dice": delta,
            "id": ids
        })
    
        # Sort by delta
        df = df.sort_values("delta_dice")
    
        # Assign effect
        df["effect"] = np.where(
            df["delta_dice"] > 0,
            "Decrease (Worse)",
            "Increase (Better)"
        )
    
        # Color map
        color_map = {
            "Decrease (Worse)": "red",
            "Increase (Better)": "green"
        }
    
        # Plot
        fig = px.bar(
            df,
            x="index",
            y="delta_dice",
            color="effect",
            color_discrete_map=color_map,
            hover_data=["id", "delta_dice"],
            labels={
                "index": xlabel,
                "delta_dice": ylabel,
                "effect": "Effect"
            },
            title=title
        )
    
        # Zero reference
        fig.add_hline(y=0)
    
        # Layout
        fig.update_layout(
            title_x=0.5,
            width=900,
            height=600,
            legend_title_text="Perturbation Effect"
        )
    
        if show:
            fig.show()
            return None
    
    
        return fig

    @staticmethod
    def dice_boxplot_under_perturbations_interactive(
        data=None,
        values=None,
        ids=None,
        perturbations=None,
        dice_columns=None,
        title="Dice Distribution under Perturbations",
        x_label="Perturbation",
        y_label="Dice Score",
        jitter=0.12,
        point_size=6,
        box_width=0.5,
        show=True
        ):
        """
        Interactive boxplot of Dice scores under different perturbations.
    
        If a DataFrame is provided and contains 'image_id',
        it is automatically used for hover labels.
    
        Parameters
        ----------
        data : pd.DataFrame, optional
        values : array-like, optional
        ids : array-like, optional
        perturbations : array-like, optional
        dice_columns : list, optional
        show : bool
    
        Returns
        -------
        fig : plotly.graph_objects.Figure or None
        """
    
        import numpy as np
        import plotly.graph_objects as go
    
        # =====================================================
        # 1. DataFrame Mode
        # =====================================================
    
        if data is not None:
    
            # Auto-detect Dice columns
            if dice_columns is None:
                dice_columns = [
                    c for c in data.columns
                    if c.lower().startswith("dice")
                ]
    
            if len(dice_columns) == 0:
                raise ValueError("No Dice columns found.")
    
            # Convert to long format
            df_long = data.melt(
                id_vars=["image_id"] if "image_id" in data.columns else None,
                value_vars=dice_columns,
                var_name="Perturbation",
                value_name="Dice"
            )
    
            # Clean labels
            df_long["Perturbation"] = (
                df_long["Perturbation"]
                .str.replace("dice_", "", case=False)
                .str.replace("_", " ")
                .str.title()
            )
    
            # Extract arrays
            values = df_long["Dice"].values
            perturbations = df_long["Perturbation"].values
    
            # Use image_id if available
            if "image_id" in df_long.columns:
                ids = df_long["image_id"].astype(str).values
            else:
                ids = df_long.index.astype(str).values
    
        # =====================================================
        # 2. Array Mode
        # =====================================================
    
        else:
    
            values = np.asarray(values, dtype=float)
            perturbations = np.asarray(perturbations, dtype=str)
            ids = np.asarray(ids, dtype=str)
    
            if not (len(values) == len(perturbations) == len(ids)):
                raise ValueError("values, ids, perturbations must match in length.")
    
        # =====================================================
        # 3. Groups
        # =====================================================
    
        unique_groups = np.unique(perturbations)
    
        group_positions = {
            grp: float(i) for i, grp in enumerate(unique_groups)
        }
    
        fig = go.Figure()
    
        # =====================================================
        # 4. Box Traces
        # =====================================================
    
        for grp in unique_groups:
    
            mask = perturbations == grp
            grp_values = values[mask]
    
            x_pos = group_positions[grp]
    
            fig.add_trace(go.Box(
                x=np.full(len(grp_values), x_pos),
                y=grp_values,
                name=str(grp),
                boxpoints="outliers",
                marker=dict(size=point_size),
                boxmean=False,
                width=box_width,
                showlegend=False,
                hoverinfo="skip"
            ))
    
        # =====================================================
        # 5. Scatter (Hover IDs)
        # =====================================================
    
        for grp in unique_groups:
    
            mask = perturbations == grp
    
            grp_values = values[mask]
            grp_ids = ids[mask]
    
            x_center = group_positions[grp]
    
            jittered_x = (
                x_center +
                (np.random.rand(len(grp_values)) - 0.5) * jitter
            )
    
            fig.add_trace(go.Scatter(
                x=jittered_x,
                y=grp_values,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color="rgba(0,0,0,0.6)"
                ),
                text=[
                    f"<b>Image:</b> {pid}"
                    f"<br><b>Dice:</b> {val:.4f}"
                    for pid, val in zip(grp_ids, grp_values)
                ],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            ))
    
        # =====================================================
        # 6. Layout
        # =====================================================
    
        fig.update_layout(
    
            title=dict(text=title, x=0.5),
    
            xaxis=dict(
                tickmode="array",
                tickvals=list(group_positions.values()),
                ticktext=list(group_positions.keys()),
                title=x_label
            ),
    
            yaxis=dict(title=y_label),
    
            template="plotly_white",
    
            margin=dict(l=60, r=30, t=60, b=60),
    
            hoverlabel=dict(bgcolor="white", font_size=13)
        )
    
        # =====================================================
        # 7. Show / Return
        # =====================================================
    
        if show:
            fig.show()
            return None
    
        return fig







