import pandas as pd
from typing import List


def join_align(peaks: pd.DataFrame, mz_tol: float, rt_tol: float) -> pd.DataFrame:
    """Align peaks across samples.

    Parameters
    ----------
    peaks: DataFrame
        Input peaks with at least ``mz``, ``rt``, ``intensity`` and ``sample``
        columns.
    mz_tol: float
        Absolute mass tolerance used when grouping peaks.
    rt_tol: float
        Retention time tolerance used when grouping peaks.

    Returns
    -------
    DataFrame
        Aligned intensity matrix with mean ``mz`` and ``rt`` for each group and
        a column for each sample.
    """
    required = {"mz", "rt", "intensity", "sample"}
    if not required.issubset(peaks.columns):
        missing = required - set(peaks.columns)
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    peaks = peaks.sort_values(["mz", "rt"]).reset_index(drop=True)

    groups: List[dict] = []
    for row in peaks.to_dict("records"):
        assigned = False
        for grp in groups:
            if abs(row["mz"] - grp["mz"]) <= mz_tol and abs(row["rt"] - grp["rt"]) <= rt_tol:
                grp["members"].append(row)
                # update mean mz/rt
                grp["mz"] = sum(p["mz"] for p in grp["members"]) / len(grp["members"])
                grp["rt"] = sum(p["rt"] for p in grp["members"]) / len(grp["members"])
                assigned = True
                break
        if not assigned:
            groups.append({"mz": row["mz"], "rt": row["rt"], "members": [row]})

    samples = sorted(peaks["sample"].unique())
    aligned_rows = []
    for grp in groups:
        out = {"mz": grp["mz"], "rt": grp["rt"]}
        for s in samples:
            ints = [p["intensity"] for p in grp["members"] if p["sample"] == s]
            out[s] = sum(ints) if ints else 0.0
        aligned_rows.append(out)
    return pd.DataFrame(aligned_rows)
