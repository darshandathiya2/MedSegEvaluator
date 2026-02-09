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







