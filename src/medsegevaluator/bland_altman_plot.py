import numpy as np
import matplotlib.pyplot as plt

def bland_altman_plot(values1, values2, title="Bland–Altman Plot", units=None, figsize=(6, 6)):
    """
    Create a Bland–Altman plot between two sets of measurements.

    Args:
        values1 (array-like): Measurements from method 1 (e.g., ground truth volumes).
        values2 (array-like): Measurements from method 2 (e.g., predicted volumes).
        title (str): Plot title.
        units (str): Label for y-axis (e.g., "mm³").
        figsize (tuple): Figure size.

    Returns:
        mean_diff (float): Mean difference between the two sets.
        loa_upper (float): Upper limit of agreement.
        loa_lower (float): Lower limit of agreement.
    """
    values1 = np.asarray(values1, dtype=float)
    values2 = np.asarray(values2, dtype=float)

    # Mean and difference
    mean_values = (values1 + values2) / 2
    diff = values1 - values2

    # Compute statistics
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff)
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(mean_values, diff, color="steelblue", alpha=0.7, edgecolor="k")
    plt.axhline(mean_diff, color="red", linestyle="--", label=f"Mean Diff = {mean_diff:.2f}")
    plt.axhline(loa_upper, color="green", linestyle="--", label=f"+1.96 SD = {loa_upper:.2f}")
    plt.axhline(loa_lower, color="orange", linestyle="--", label=f"-1.96 SD = {loa_lower:.2f}")
    plt.xlabel("Mean of Methods" + (f" ({units})" if units else ""))
    plt.ylabel("Difference (Method1 - Method2)" + (f" ({units})" if units else ""))
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.show()

    return mean_diff, loa_upper, loa_lower
