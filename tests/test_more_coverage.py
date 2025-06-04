import os
import sys
import runpy
import numpy as np
import pytest

from vimms.Column import CleanColumn, LinearColumn
from vimms.Controller.noise import ThresholdEstimator, IncreaseEstimator
from vimms.scripts.openms_optimise_params import ParametersBuilder, TopNParameters
from pathlib import Path
import vimms

remove_lines_path = Path(vimms.__file__).resolve().parent / 'scripts' / 'remove_lines.py'
from vimms.MassSpecUtils import adduct_transformation
from vimms.Utils import decimal_to_string


class FakeChem:
    def __init__(self, rt, apex):
        self.rt = rt
        self._apex = apex
    def get_apex_rt(self):
        return self._apex


class FakeROI:
    def __init__(self, intensities, apex):
        self.intensity_list = intensities
        self._apex = apex
    def estimate_apex(self):
        return self._apex


def test_clean_column_no_noise():
    chems = [FakeChem(i, i) for i in range(3)]
    col = CleanColumn(chems)
    new = col.get_dataset()
    assert [c.rt for c in new] == [0, 1, 2]


def test_linear_column_fixed_offsets():
    chems = [FakeChem(0, a) for a in (0, 1, 2)]
    lc = LinearColumn.from_fixed_offsets(chems, 0.0, 1.0, 2.0)
    # offsets = intercept + linear * apex_rt
    expected = [1.0 + 2.0 * a for a in (0, 1, 2)]
    assert np.allclose(lc.offsets, expected)


def test_linear_column_drift_fn():
    roi = FakeROI([1, 2, 3], 10)
    lc = LinearColumn.from_fixed_offsets([FakeChem(0, 0)], 0.0, 1.0, 1.0)
    drift, extra = lc.drift_fn(roi, 0)
    assert drift == pytest.approx(10 - (10 - 1.0) / 2)
    assert extra == {}


def test_threshold_and_increase_estimators():
    roi = FakeROI([0, 5, 6, 7], 0)
    t = ThresholdEstimator(intensity_threshold=5, count_threshold=2)
    assert t.estimate_noise(roi) == 1.0
    i = IncreaseEstimator(count_threshold=2)
    assert i.estimate_noise(roi) == 1.0


def test_parameters_builder_set_and_error():
    pb = ParametersBuilder(TopNParameters)
    pb.set('N', 20)
    params = pb.build()
    assert params.N == 20
    with pytest.raises(ValueError):
        pb.set('NOT_REAL', 1)


def test_remove_lines_script(tmp_path, monkeypatch):
    f = tmp_path / 'file.txt'
    f.write_text('a\nb\nc\nd\ne\nf\n')
    monkeypatch.setattr(sys, 'argv', ['remove_lines.py', str(f)])
    runpy.run_path(remove_lines_path, run_name='__main__')
    assert f.read_text().splitlines() == ['d', 'f']


def test_decimal_to_string_and_adduct_transformation():
    assert decimal_to_string(4.5) == '4'
    assert decimal_to_string(4.567, 2) == '4.57'
    assert adduct_transformation(100.0, 1.0, 2.0) == 102.0
