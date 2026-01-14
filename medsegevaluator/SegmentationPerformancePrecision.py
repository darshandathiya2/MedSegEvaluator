from __future__ import annotations

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date
from typing import Callable, Tuple


class SegmentationPerformancePrecision:
    r"""
    SegmentationPerformancePrecision
    =================================

    A statistical utility class for **quantifying the precision and reliability**
    of medical image segmentation model performance metrics.

    This class evaluates how **stable, repeatable, and well-estimated**
    segmentation metrics (e.g., Dice, HD95, NSD) are when computed on finite
    test datasets.


    Specifically, the class enables:
    - Analytical precision estimation using mean, standard deviation, SEM, and confidence interval width
    - Bootstrap-based precision analysis to assess metric stability
    - Evaluation of performance reliability as a function of sample size
    - Slice-wise, patient-wise, and cohort-level precision analysis

    This module is intended for rigorous and reproducible reporting of
    segmentation model performance in medical imaging studies.
    """

    # ------------------------------------------------------------------
    # Basic statistical analysis
    # ------------------------------------------------------------------
    @staticmethod
    def statistical_analysis(df: pd.DataFrame):
        """
        Compute analytical statistics for a metric sample.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing metric values.

        Returns
        -------
        mean : float
            Sample mean
        std : float
            Sample standard deviation
        SEM : float
            Standard Error of the Mean
        w : float
            Width of 95% confidence interval (2 * 1.96 * SEM)
        """
        mean = np.round(df.mean(), 4)
        std = np.round(df.std(), 4)
        SEM = np.round(std / np.sqrt(len(df)), 4)
        w = 2 * 1.96 * SEM

        print(str(mean) + "&" + str(std) + "&" + str(SEM) + "&" + str(w))
        return mean, std, SEM, w

    # ------------------------------------------------------------------
    # Bootstrap replicate generation
    # ------------------------------------------------------------------
    @staticmethod
    def draw_bs_replicates(
        data: np.ndarray,
        func: Callable,
        size: int
    ) -> np.ndarray:
        r"""
        Generate bootstrap replicates for a statistic.

        Parameters
        ----------
        data : np.ndarray
            Input metric values
        func : Callable
            Statistic to compute (e.g., np.mean)
        size : int
            Number of bootstrap replicates

        Returns
        -------
        np.ndarray
            Bootstrap replicates
        """
        bs_replicates = np.empty(size)

        for i in range(size):
            bs_sample = np.random.choice(data, size=len(data), replace=True)
            bs_replicates[i] = func(bs_sample)

        return bs_replicates

    # ------------------------------------------------------------------
    # Bootstrap-based precision analysis
    # ------------------------------------------------------------------
    @staticmethod
    def bootstrap_analysis(
        data: pd.Series | pd.DataFrame,
        k: int | None = None,
        n_replicates: int = 15000
    ) -> Tuple[float, float, float, np.ndarray]:
        r"""
        Perform bootstrap-based precision analysis.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Metric values
        k : int, optional
            Sub-sample size (if None, use full dataset)
        n_replicates : int
            Number of bootstrap replicates

        Returns
        -------
        mu_star : float
            Bootstrap mean
        sem_star : float
            Bootstrap standard error
        w_star : float
            Bootstrap confidence interval width
        conf_interval : np.ndarray
            95% bootstrap confidence interval
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name="metric")

        if k is None:
            k = len(data)

        # Sub-sampling without replacement
        metric_samples = data["metric"].sample(k, replace=False).reset_index(drop=True)

        bs_replicates = SegmentationPerformancePrecision.draw_bs_replicates(
            metric_samples.values,
            np.mean,
            n_replicates
        )

        conf_interval = np.percentile(bs_replicates, [2.5, 97.5]).round(4)
        w_star = conf_interval[1] - conf_interval[0]
        mu_star = np.mean(bs_replicates)
        sem_star = np.std(bs_replicates)

        return mu_star, sem_star, w_star, conf_interval

    # ------------------------------------------------------------------
    # Sub-sampling experiment for precision vs sample size
    # ------------------------------------------------------------------
    @staticmethod
    def create_subsampling_data(
        K_samples: list[int],
        data: pd.DataFrame,
        save_dir: str,
        experiment_name: str,
        metric_name: str,
        n_trials: int = 100
    ) -> str:
        r"""
        Conduct repeated sub-sampling experiments to analyze
        precision vs sample size.

        Parameters
        ----------
        K_samples : list[int]
            Sub-sample sizes (e.g., [10, 20, 50, 100])
        data : pd.DataFrame
            DataFrame with column 'metric'
        save_dir : str
            Directory to save CSV output
        experiment_name : str
            Name of experiment (e.g., model/dataset)
        metric_name : str
            Metric name (e.g., Dice, HD95)
        n_trials : int
            Number of repetitions per K

        Returns
        -------
        str
            Absolute path to saved CSV file
        """

        records = []

        for k in K_samples:
            for _ in tqdm(range(n_trials), desc=f"K={k}"):
                sample = data["metric"].sample(k, replace=False)

                # Analytical statistics
                mean_k = np.mean(sample)
                std_k = np.std(sample)
                sem_k = std_k / np.sqrt(k)
                w_k = 2 * 1.96 * sem_k
                gci = (
                    mean_k - 1.96 * sem_k,
                    mean_k + 1.96 * sem_k
                )

                # Bootstrap statistics
                mu_star, sem_star, w_star, bci = (
                    SegmentationPerformancePrecision.bootstrap_analysis(sample)
                )

                records.append({
                    "sample_size": k,
                    "mean": mean_k,
                    "std": std_k,
                    "SEM": sem_k,
                    "CI_width": w_k,
                    "GCI_low": gci[0],
                    "GCI_high": gci[1],
                    "bootstrap_mean": mu_star,
                    "bootstrap_SEM": sem_star,
                    "bootstrap_CI_width": w_star,
                    "BCI_low": bci[0],
                    "BCI_high": bci[1],
                })

        df = pd.DataFrame(records)

        today = date.today().strftime("%b%d")
        filename = f"precision-analysis-{metric_name}-{experiment_name}-{today}.csv"
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)

        return filepath
