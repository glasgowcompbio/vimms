import pandas as pd
from demo.untargeted.join_aligner import join_align


def test_single_group():
    df = pd.DataFrame(
        {
            "mz": [100.0, 100.01],
            "rt": [1.0, 1.02],
            "intensity": [10, 20],
            "sample": ["A", "B"],
        }
    )
    out = join_align(df, mz_tol=0.05, rt_tol=0.05)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["A"] == 10
    assert row["B"] == 20


def test_two_groups():
    df = pd.DataFrame(
        {
            "mz": [100.0, 100.04, 100.5],
            "rt": [1.0, 1.02, 1.01],
            "intensity": [5, 7, 9],
            "sample": ["A", "B", "A"],
        }
    )
    out = join_align(df, mz_tol=0.05, rt_tol=0.05)
    assert len(out) == 2
    # group 0 should contain first two peaks
    first = out.iloc[0]
    second = out.iloc[1]
    if first["A"] == 5:
        assert first["B"] == 7
        assert second["A"] == 9
    else:
        assert first["A"] == 9
        assert second["A"] == 5
        assert second["B"] == 7
