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
) -> dict:
    """Run the full untargeted demo pipeline and return alignment metrics."""

    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate chemicals and dataset design
    chemicals, design = setup_simulation(n_chemicals, n_samples_per_group)

    # Generate mzML files
    generate_mzml_files(chemicals, design, out_dir, max_rt=max_rt, top_n=top_n)

    # Ground truth table and library
    gt = generate_ground_truth_table(chemicals, design)
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

    metrics = run_pipeline()
    report_metrics(metrics)


if __name__ == "__main__":
    main()
