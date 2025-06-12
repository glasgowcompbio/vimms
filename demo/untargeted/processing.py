from __future__ import annotations

"""Stub processing steps for the untargeted workflow."""

from pathlib import Path
import json
import pandas as pd

from .generate_dataset import Dataset, ExperimentalDesign
from .evaluation import compute_group_metrics
from .peak_picking import peak_table_from_ground_truth


def get_peak_data(gt: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """Return peaks derived from ``gt`` and optionally write to ``out_dir``.

    Parameters
    ----------
    gt:
        Ground truth table describing the true peaks.
    out_dir:
        Directory where ``peaks.csv`` will be written when provided.

    Returns
    -------
    pd.DataFrame
        Peak table containing ``sample``, ``mz``, ``rt`` and ``intensity``.
    """
    peaks = peak_table_from_ground_truth(gt)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        peaks.to_csv(out_dir / "peaks.csv", index=False)
    return peaks


class OutputWriter:
    """Utility to persist pipeline outputs to disk."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        peaks: pd.DataFrame | None = None,
        aligned: pd.DataFrame | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Write provided outputs to :pyattr:`out_dir`."""

        if peaks is not None:
            peaks.to_csv(self.out_dir / "peaks.csv", index=False)
        if aligned is not None:
            aligned.to_csv(self.out_dir / "aligned.csv", index=False)
        if metrics is not None:
            (self.out_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )


def group_related_peaks(aligned: pd.DataFrame) -> pd.DataFrame:
    """Group isotopes and adducts together.

    Parameters
    ----------
    aligned:
        Aligned feature table with sample intensities.

    Returns
    -------
    pd.DataFrame
        Table of grouped peaks. Currently returned unchanged.
    """
    return aligned


def identify_compounds(
    grouped: pd.DataFrame, library_file: Path | None = None
) -> pd.DataFrame:
    """Identify compounds by matching grouped peaks to a library.

    Parameters
    ----------
    grouped:
        Peak table after grouping related peaks.
    library_file:
        Optional path to an MGF or MSP library for matching.

    Returns
    -------
    pd.DataFrame
        Peak table with identification columns added. Returned unchanged for now.
    """
    return grouped


def annotate_spectra(
    identified: pd.DataFrame, library_file: Path | None = None
) -> pd.DataFrame:
    """Add spectral annotations to identified features.

    Parameters
    ----------
    identified:
        Peak table containing compound identifications.
    library_file:
        Optional path to a spectral library.

    Returns
    -------
    pd.DataFrame
        Annotated peak table. Returned unchanged for now.
    """
    return identified


def batch_normalize(peaks: pd.DataFrame, design: ExperimentalDesign) -> pd.DataFrame:
    """Correct systematic effects between sample groups or batches.

    Parameters
    ----------
    peaks:
        Table of peak intensities.
    design:
        Experimental design describing sample groups.

    Returns
    -------
    pd.DataFrame
        Normalized peak table. Returned unchanged for now.
    """
    return peaks


def impute_missing_values(peaks: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing intensity values.

    Parameters
    ----------
    peaks:
        Table of peak intensities.

    Returns
    -------
    pd.DataFrame
        Peak table with missing values imputed. Returned unchanged for now.
    """
    return peaks


def compute_statistics(labeled: pd.DataFrame, dataset: Dataset) -> dict:
    """Compute evaluation metrics for the aligned peaks.

    Parameters
    ----------
    labeled:
        Peak-level table with grouping labels from the aligner.
    dataset:
        Dataset object containing the ground-truth table.

    Returns
    -------
    dict
        Dictionary of computed metrics. Empty if ground truth is unavailable.
    """
    if dataset.ground_truth is None:
        return {}

    gt_eval = dataset.ground_truth[
        ["sample", "compound_id", "mz_apex", "rt_apex", "intensity"]
    ].rename(columns={"mz_apex": "mz", "rt_apex": "rt"})
    df_eval = labeled.merge(
        gt_eval, on=["sample", "mz", "rt", "intensity"], how="left"
    )
    metrics = compute_group_metrics(df_eval, compound_col="compound_id", group_col="group")
    return metrics


def generate_report(metrics: dict) -> str:
    """Return a JSON report summarizing ``metrics``."""

    return json.dumps(metrics, indent=2)
