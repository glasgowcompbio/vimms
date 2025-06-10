from __future__ import annotations

import json
from pathlib import Path


from .generate_dataset import (
    setup_simulation,
    generate_mzml_files,
    generate_ground_truth_table,
    write_ground_truth_mgf,
)
from .peak_picking import peak_table_from_ground_truth
from .join_aligner import join_align
from .evaluation import compute_group_metrics


def report_metrics(metrics: dict) -> None:
    """Print alignment metrics in a friendly format."""

    print("Alignment metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def run_pipeline(
    out_dir: Path = Path("./out"),
    n_chemicals: int = 100,
    n_samples_per_group: int = 5,
    mz_tol: float = 0.01,
    rt_tol: float = 0.5,
    max_rt: int = 180,
    top_n: int = 1,
    use_rt_noise: bool = False,
    noise_sd: float = 0.1,
    intercept_params: tuple[float, float] = (0.0, 5.0),
    linear_params: tuple[float, float] = (0.0, 0.001),
) -> dict:
    """Run the full untargeted demo pipeline and return alignment metrics.

    Parameters
    ----------
    use_rt_noise:
        Whether to apply retention time drift to each injection using a
        :class:`~vimms.ColumnDrift.SimulatedDriftModel`.
    noise_sd:
        Standard deviation of random noise added around the drift function.
    intercept_params:
        Mean and standard deviation of the intercept term (seconds).
    linear_params:
        Mean and standard deviation of the linear term.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate chemicals and dataset design
    chemicals, design = setup_simulation(n_chemicals, n_samples_per_group)

    column_params = None
    if use_rt_noise:
        column_params = {
            "noise_sd": noise_sd,
            "intercept_params": intercept_params,
            "linear_params": linear_params,
        }

    # Generate mzML files
    sample_chems = generate_mzml_files(
        chemicals,
        design,
        out_dir,
        max_rt=max_rt,
        top_n=top_n,
        column_params=column_params,
    )

    # Ground truth table and library
    gt = generate_ground_truth_table(
        chemicals,
        design,
        per_sample_chems=sample_chems if use_rt_noise else None,
    )
    gt_file = out_dir / "ground_truth.csv"
    gt.to_csv(gt_file, index=False)
    write_ground_truth_mgf(chemicals, out_dir / "ground_truth.mgf")

    # Peak picking from ground truth
    peaks = peak_table_from_ground_truth(gt)
    peaks_file = out_dir / "peaks.csv"
    peaks.to_csv(peaks_file, index=False)

    # Alignment
    aligned, labeled = join_align(peaks, mz_tol, rt_tol, return_labels=True)
    aligned.to_csv(out_dir / "aligned.csv")

    # Join group labels back to ground truth for evaluation
    gt_eval = gt[["sample", "compound_id", "mz_apex", "rt_apex", "intensity"]].rename(
        columns={"mz_apex": "mz", "rt_apex": "rt"}
    )
    df_eval = labeled.merge(gt_eval, on=["sample", "mz", "rt", "intensity"], how="left")

    metrics = compute_group_metrics(df_eval, compound_col="compound_id", group_col="group")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    """Command-line entry point for running the pipeline."""

    metrics = run_pipeline(use_rt_noise=True)
    report_metrics(metrics)


if __name__ == "__main__":
    main()
