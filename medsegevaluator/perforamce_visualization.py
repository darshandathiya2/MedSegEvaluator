import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from skimage import measure
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




