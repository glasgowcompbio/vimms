import pandas as pd
from pathlib import Path


def peak_table_from_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    """Return a simple peak table derived from the ground truth.

    Parameters
    ----------
    gt : pd.DataFrame
        Ground truth table produced by :func:`generate_ground_truth_table`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``sample``, ``mz``, ``rt`` and ``intensity``.
    """
    required = {"sample", "mz_apex", "rt_apex", "intensity"}
    if not required <= set(gt.columns):
        raise ValueError(f"Ground truth must contain columns {required}")
    return gt.rename(columns={"mz_apex": "mz", "rt_apex": "rt"})[
        ["sample", "mz", "rt", "intensity"]
    ]


def write_peak_table(gt_file: Path, out_file: Path) -> pd.DataFrame:
    """Read ``gt_file`` and write the peak table to ``out_file``."""
    gt = pd.read_csv(gt_file)
    peaks = peak_table_from_ground_truth(gt)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(out_file, index=False)
    return peaks


def main() -> None:
    """Entry point for the peak picking step."""
    gt_file = Path("./out/ground_truth.csv")
    out_file = Path("./out/peaks.csv")
    write_peak_table(gt_file, out_file)


if __name__ == "__main__":
    main()
