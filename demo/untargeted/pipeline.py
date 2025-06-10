from pathlib import Path

from .generate_dataset import Dataset, generate_synthetic_dataset
from .join_aligner import join_align
from .processing import (
    get_peak_data,
    OutputWriter,
    group_related_peaks,
    identify_compounds,
    annotate_spectra,
    batch_normalize,
    impute_missing_values,
    compute_statistics,
    generate_report,
)


def report_metrics(metrics):
    """Print alignment metrics in a friendly format."""

    print("Alignment metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def run_pipeline(dataset: Dataset, out_dir: Path, mz_tol: float = 0.01, rt_tol: float = 0.5) -> dict | None:
    """Run the preprocessing pipeline on ``dataset`` and return metrics if available."""

    writer = OutputWriter(out_dir)

    if dataset.ground_truth is None:
        raise ValueError("Ground truth is required for the demo pipeline")

    peaks = get_peak_data(dataset.ground_truth)

    aligned, labeled = join_align(peaks, mz_tol, rt_tol, return_labels=True)

    grouped = group_related_peaks(aligned)
    identified = identify_compounds(grouped, dataset.mgf_file)
    annotated = annotate_spectra(identified, dataset.mgf_file)
    normalized = batch_normalize(annotated, dataset.design)
    _ = impute_missing_values(normalized)

    metrics = compute_statistics(labeled, dataset)
    writer.write_all(peaks=peaks, aligned=aligned, metrics=metrics)
    _ = generate_report(metrics)
    return metrics


def main() -> None:
    """Command-line entry point for running the pipeline."""
    dataset = generate_synthetic_dataset(Path("./out"), use_rt_noise=True)
    metrics = run_pipeline(dataset, Path("./out"))
    report_metrics(metrics)


if __name__ == "__main__":
    main()
