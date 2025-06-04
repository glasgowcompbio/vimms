import pytest
import csv
import numpy as np
import pandas as pd
from vimms.DIA import DiaWindows
from vimms.Controller.targeted import Target, create_targets_from_toxid
from vimms.Evaluation import Evaluator
from vimms.Controller.misc import TaskFilter
from vimms.Common import get_dda_scan_param, get_default_scan_params
from vimms.scripts.check_ms2_matches import (
    extract_msdial_spectrum, msdial_row_to_box, chem_to_spectral_record
)


# ---- DIA tests ----

def test_dia_windows_basic_even():
    ms1_mzs = np.linspace(0, 100, 11)
    dw = DiaWindows(
        ms1_mzs,
        ms1_range=[(0, 100)],
        dia_design="basic",
        window_type="even",
        kaufmann_design=None,
        extra_bins=0,
        num_windows=4,
    )
    internal = [-1, 25, 50, 75, 101]
    for i in range(4):
        assert dw.locations[i][0][0] == (
            internal[i], internal[i + 1]
        )


def test_dia_windows_invalid():
    with pytest.raises(ValueError):
        DiaWindows(np.arange(5), [(0, 1)], "basic", "even", None, extra_bins=1, num_windows=2)


# ---- Targeted tests ----

def test_target_peak_and_active():
    t = Target(100, 99, 101, 10, 20, name="A")
    assert t.peak_in(100, 15)
    assert not t.peak_in(102, 15)
    mz_int = [(100, 1000)]
    assert t.active(mz_int, 15, 500)
    assert not t.active(mz_int, 5, 500)


def test_target_str():
    t = Target(100, 99, 101, 10, 20, name="A", adduct="[M+H]+")
    assert "A[M+H]+" in str(t)


def test_create_targets_from_toxid(tmp_path):
    path = tmp_path / "toxid.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Header"])
        writer.writerow(["Index"])
        writer.writerow([1, "Water", "H2O", "+", "x", 1.0, "", "", ""])
    targets = create_targets_from_toxid(path, adducts_to_use=["[M+H]+"])
    assert len(targets) == 1
    t = targets[0]
    assert t.metadata["name"] == "Water"
    assert pytest.approx(t.from_rt) == 0.0
    assert pytest.approx(t.to_rt) == 120.0


# ---- Evaluation tests ----

def test_new_window_interval():
    i = Evaluator._new_window(5, 100, 2)
    assert i.pt1.x == i.pt2.x == 5
    assert i.pt1.y == 99
    assert i.pt2.y == 101


# ---- TaskFilter tests ----

def _make_ms1(ms1_id):
    sp = get_default_scan_params(scan_id=ms1_id)
    return sp


def _make_ms2(scan_id, precursor_id):
    return get_dda_scan_param(100, 1e5, precursor_id, 1, 10, 10, scan_id=scan_id)


def test_find_nearest_ms2():
    tasks = [_make_ms1(1), _make_ms2(2, 1), _make_ms1(3)]
    tf = TaskFilter(0.1, 0.1)
    assert TaskFilter._find_nearest_ms2(0, tasks) == tasks[1]
    assert TaskFilter._find_nearest_ms2(2, tasks) == tasks[1]
    assert TaskFilter._find_nearest_ms2(1, tasks) is None


# ---- check_ms2_matches utilities ----

def test_extract_msdial_spectrum_and_box():
    row = pd.Series({"MS/MS spectrum": "100:10 110:20"})
    peaks = extract_msdial_spectrum(row, "MS/MS spectrum")
    assert np.allclose(peaks[0], [100.0, 10.0])
    box = msdial_row_to_box(100, 1, 2)
    assert box.pt1.x == 60.0 and box.pt2.x == 120.0


def test_chem_to_spectral_record():
    class FakeFrag:
        def __init__(self, mz, prop_ms2_mass, parent):
            self.isotopes = [(mz, None, "MSN")]
            self.prop_ms2_mass = prop_ms2_mass
            self.parent = parent
    class FakeChem:
        def __init__(self):
            self.isotopes = [(50.0, None, "MS1")]
            self.rt = 10.0
            self.max_intensity = 1000.0
            self.children = [FakeFrag(60.0, 0.5, self)]
    spec = chem_to_spectral_record(FakeChem())
    assert spec.precursor_mz > 50.0
    assert len(spec.peaks) == 1
    assert spec.metadata["rt"] == 10.0
