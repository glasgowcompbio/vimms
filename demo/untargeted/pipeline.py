from pathlib import Path
from abc import ABC, abstractmethod
import argparse

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


class BasePipeline(ABC):
    """Base class encapsulating preprocessing logic."""

    def __init__(
        self,
        dataset: Dataset,
        out_dir: Path,
        mz_tol: float = 0.01,
        rt_tol: float = 0.5,
    ) -> None:
        self.dataset = dataset
        self.out_dir = out_dir
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol
        self.writer = OutputWriter(out_dir)

    # ------------------------------------------------------------------
    # Steps in the preprocessing pipeline. These methods wrap the helper
    # functions defined in :mod:`processing` so subclasses can override
    # individual steps when needed.
    # ------------------------------------------------------------------

    def group_peaks(self, aligned):
        """Group isotopes/adducts together."""

        return group_related_peaks(aligned)

    def identify(self, grouped):
        """Match grouped peaks to library entries."""

        return identify_compounds(grouped, self.dataset.mgf_file)

    def annotate(self, identified):
        """Annotate identified peaks with spectral information."""

        return annotate_spectra(identified, self.dataset.mgf_file)

    def normalize(self, annotated):
        """Apply batch normalisation."""

        return batch_normalize(annotated, self.dataset.design)

    def impute(self, normalized):
        """Fill in missing values in the peak table."""

        return impute_missing_values(normalized)

    def run(self) -> dict | None:
        """Run the preprocessing pipeline."""

        peaks = self.get_peaks()
        aligned, labeled = join_align(
            peaks, self.mz_tol, self.rt_tol, return_labels=True
        )

        grouped = self.group_peaks(aligned)
        identified = self.identify(grouped)
        annotated = self.annotate(identified)
        normalized = self.normalize(annotated)
        _ = self.impute(normalized)

        metrics = compute_statistics(labeled, self.dataset)
        self.writer.write_all(peaks=peaks, aligned=aligned, metrics=metrics)
        _ = generate_report(metrics)
        return metrics

    @abstractmethod
    def get_peaks(self):
        """Return the peak table for alignment."""


class SyntheticPipeline(BasePipeline):
    """Pipeline that expects ground-truth derived peaks."""

    def get_peaks(self):
        if self.dataset.ground_truth is None:
            raise ValueError("Ground truth is required for the demo pipeline")
        return get_peak_data(self.dataset.ground_truth)


def report_metrics(metrics):
    """Print alignment metrics in a friendly format."""

    print("Alignment metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def run_pipeline(
    dataset: Dataset, out_dir: Path, mz_tol: float = 0.01, rt_tol: float = 0.5
) -> dict | None:
    """Convenience wrapper executing the synthetic pipeline."""

    pipeline = SyntheticPipeline(
        dataset=dataset, out_dir=out_dir, mz_tol=mz_tol, rt_tol=rt_tol
    )
    return pipeline.run()


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for generating data and running the pipeline."""

    parser = argparse.ArgumentParser(description="Run the untargeted demo pipeline")
    parser.add_argument("--out-dir", type=Path, default=Path("./out"))
    parser.add_argument("--use-rt-noise", action="store_true", help="Add RT noise to the synthetic dataset")
    parser.add_argument("--mz-tol", type=float, default=0.01)
    parser.add_argument("--rt-tol", type=float, default=0.5)
    args = parser.parse_args(argv)

    dataset = generate_synthetic_dataset(args.out_dir, use_rt_noise=args.use_rt_noise)
    metrics = run_pipeline(dataset, args.out_dir, mz_tol=args.mz_tol, rt_tol=args.rt_tol)
    report_metrics(metrics)


if __name__ == "__main__":
    main()
