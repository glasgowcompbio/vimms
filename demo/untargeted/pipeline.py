import json
from pathlib import Path

from .dataset import UntargetedDataset, generate_synthetic_dataset
from .peak_picking import peak_table_from_ground_truth
from .join_aligner import join_align
from .evaluation import compute_group_metrics


def report_metrics(metrics: dict) -> None:
    """Print alignment metrics in a friendly format."""

    print("Alignment metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def run_pipeline(
    dataset: UntargetedDataset,
    out_dir: Path = Path("./out"),
    mz_tol: float = 0.01,
    rt_tol: float = 0.5,
) -> dict:
    """Run the untargeted demo pipeline on ``dataset``."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if dataset.ground_truth is None:
        raise ValueError("Ground truth table required for this demo")

    peaks = peak_table_from_ground_truth(dataset.ground_truth)
    peaks_file = out_dir / "peaks.csv"
    peaks.to_csv(peaks_file, index=False)

    aligned, labeled = join_align(peaks, mz_tol, rt_tol, return_labels=True)
    aligned.to_csv(out_dir / "aligned.csv")

    gt_eval = dataset.ground_truth[
        ["sample", "compound_id", "mz_apex", "rt_apex", "intensity"]
    ].rename(columns={"mz_apex": "mz", "rt_apex": "rt"})
    df_eval = labeled.merge(gt_eval, on=["sample", "mz", "rt", "intensity"], how="left")

    metrics = compute_group_metrics(df_eval, compound_col="compound_id", group_col="group")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    """Command-line entry point for running the pipeline."""

    dataset = generate_synthetic_dataset(Path("./out"))
    metrics = run_pipeline(dataset, Path("./out"))
    report_metrics(metrics)


if __name__ == "__main__":
    main()
