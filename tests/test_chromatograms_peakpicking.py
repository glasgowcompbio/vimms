import pytest
import os
import numpy as np
from vimms.Chromatograms import EmpiricalChromatogram, ConstantChromatogram, FunctionalChromatogram
from vimms.PeakPicking import count_boxes, format_output_path, MZMineParams


def test_empirical_chromatogram_sort_and_normalise():
    rts = np.array([2.0, 1.0, 3.0])
    mzs = np.array([100.0, 101.0, 102.0])
    ints = np.array([10.0, 30.0, 20.0])
    chrom = EmpiricalChromatogram(rts, mzs, ints)
    # rts should be sorted internally
    assert np.all(np.diff(chrom.rts) >= 0)
    # intensities normalised to 1
    assert chrom.intensities.max() == 1.0
    # relative intensity at first time point should match first sorted intensity
    ri = chrom.get_relative_intensity(chrom.rts[0])
    assert ri == chrom.intensities[0]


def test_empirical_chromatogram_single_point():
    chrom = EmpiricalChromatogram(np.array([5.0]), np.array([50.0]), np.array([5.0]))
    # single point gets expanded into length-two arrays
    assert len(chrom.rts) == 2
    assert chrom.intensities[0] == chrom.intensities[1]


def test_constant_chromatogram():
    chrom = ConstantChromatogram()
    assert chrom.get_relative_intensity(0.0) == 1.0
    assert chrom.get_relative_mz(10.0) == 0.0
    assert chrom._rt_match(-1.0)
    assert chrom.get_apex_rt() == 0.0


def test_functional_chromatogram_normal():
    chrom = FunctionalChromatogram("normal", [0.0, 1.0])
    # apex occurs at the midpoint of rt range
    expected_apex = (chrom.max_rt - chrom.min_rt) / 2
    assert np.isclose(chrom.get_apex_rt(), expected_apex)
    assert chrom._rt_match(0.0)
    # relative intensity at mean should be maximal
    assert chrom.get_relative_intensity(expected_apex) == pytest.approx(1.0, rel=1e-3)


def test_count_boxes(tmp_path):
    path = tmp_path / "boxes.csv"
    with open(path, "w") as f:
        f.write("header\n")
        f.write("1\n2\n3\n")
    assert count_boxes(path) == 3


def test_format_output_path(tmp_path):
    out = format_output_path("TEST", tmp_path, "sample.csv")
    assert out.endswith("sample_test_aligned.csv")
    assert os.path.dirname(out) == str(tmp_path)

    out = tmp_path / "aligned.csv"
    with open(out, "w") as f:
        f.write("file1.mzML filtered Peak Area,file2.mzML filtered Peak Area\n")
    passed, _, _ = MZMineParams.check_files_match(["file1.mzML"], out)
    passed2, _, _ = MZMineParams.check_files_match(["file1.mzML"], out, mode="exact")
    assert passed
    assert not passed2
