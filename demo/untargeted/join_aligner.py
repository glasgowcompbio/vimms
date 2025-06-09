import pandas as pd


def join_align(
    peaks: pd.DataFrame,
    mz_tol: float,
    rt_tol: float,
    return_labels: bool = False,
) -> pd.DataFrame:
    """Align peaks across samples based on m/z and RT tolerances.

    Parameters
    ----------
    peaks : pd.DataFrame
        Table containing ``sample``, ``mz``, ``rt`` and ``intensity`` columns.
    mz_tol : float
        Absolute m/z tolerance for grouping peaks.
    rt_tol : float
        Absolute retention time tolerance for grouping peaks.
    return_labels : bool, optional
        If ``True``, also return the input peaks with a ``group`` column
        describing the assigned alignment group.

    Returns
    -------
    pd.DataFrame
        An intensity table with groups as rows and samples as columns. If
        ``return_labels`` is ``True`` a tuple ``(aligned, peaks)`` is returned
        where ``peaks`` contains the assigned ``group`` for each row.
    """
    required = {"sample", "mz", "rt", "intensity"}
    if not required <= set(peaks.columns):
        raise ValueError(f"Input peaks must contain columns {required}")

    peaks = peaks.copy().sort_values(["mz", "rt"]).reset_index(drop=True)
    groups: list[dict] = []
    group_ids: list[int] = [-1] * len(peaks)

    for idx, row in peaks.iterrows():
        assigned = False
        for gid, g in enumerate(groups):
            if abs(row["mz"] - g["mz"]) <= mz_tol and abs(row["rt"] - g["rt"]) <= rt_tol:
                n = g["n"] + 1
                g["mz"] = (g["mz"] * g["n"] + row["mz"]) / n
                g["rt"] = (g["rt"] * g["n"] + row["rt"]) / n
                g["n"] = n
                group_ids[idx] = gid
                assigned = True
                break
        if not assigned:
            groups.append({"mz": row["mz"], "rt": row["rt"], "n": 1})
            group_ids[idx] = len(groups) - 1

    peaks["group"] = group_ids
    aligned = peaks.pivot_table(
        index="group",
        columns="sample",
        values="intensity",
        aggfunc="max",
        fill_value=0,
    ).sort_index()

    if return_labels:
        return aligned, peaks
    return aligned
