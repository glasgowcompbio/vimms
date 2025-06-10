
from demo.untargeted.generate_chemicals import generate_chemicals
from demo.untargeted.dataset import (
    create_design,
    generate_mzml_files,
    generate_ground_truth_table,
    write_ground_truth_mgf,
    setup_simulation,
    generate_synthetic_dataset,
)

from demo.untargeted.join_aligner import join_align
from demo.untargeted.evaluation import compute_group_metrics
from demo.untargeted.peak_picking import peak_table_from_ground_truth
from demo.untargeted.pipeline import run_pipeline
import pandas as pd


def test_generate_chemicals_length():
    chems = generate_chemicals(10)
    assert len(chems) == 10
    # ensure each chemical has expected attributes
    for chem in chems:
        assert hasattr(chem, 'mass')
        assert hasattr(chem, 'rt')


def test_create_design():
    design = create_design(3)
    assert design.samples['case'] == ['case_1', 'case_2', 'case_3']
    assert len(design.samples['control']) == 3


def test_setup_simulation_returns_data():
    chems, design = setup_simulation(20, 2)
    assert len(chems) == 20
    assert len(design.samples['case']) == 2


def test_generate_mzml_files(tmp_path):
    chems, design = setup_simulation(3, 1)
    generate_mzml_files(chems, design, tmp_path, max_rt=20, top_n=1)
    assert (tmp_path / 'case' / 'case_1.mzML').is_file()


def test_generate_ground_truth_table():
    chems, design = setup_simulation(2, 1)
    df = generate_ground_truth_table(chems, design)
    assert {'sample', 'compound_id', 'mz_min', 'mz_max'} <= set(df.columns)
    assert len(df) == 4  # 2 samples (case/control) * 2 chemicals


def test_write_ground_truth_mgf(tmp_path):
    chems = generate_chemicals(2)
    out_file = tmp_path / "lib.mgf"
    write_ground_truth_mgf(chems, out_file)
    text = out_file.read_text()
    assert text.startswith("BEGIN IONS")


def test_join_align_groups_by_tolerances():
    peaks = pd.DataFrame(
        {
            "sample": ["s1", "s1", "s2", "s3"],
            "mz": [100.0, 105.0, 100.01, 200.0],
            "rt": [1.0, 1.5, 1.01, 2.0],
            "intensity": [10, 5, 20, 30],
        }
    )
    aligned = join_align(peaks, mz_tol=0.02, rt_tol=0.05)
    assert set(aligned.columns) == {"s1", "s2", "s3"}
    assert len(aligned) == 3
    assert aligned.loc[0, "s1"] == 10
    assert aligned.loc[0, "s2"] == 20
    assert aligned.loc[1, "s1"] == 5
    assert aligned.loc[2, "s3"] == 30


def test_join_align_handles_missing_samples():
    peaks = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "mz": [50.0, 50.01],
            "rt": [0.5, 0.49],
            "intensity": [5, 15],
        }
    )
    aligned = join_align(peaks, mz_tol=0.02, rt_tol=0.02)
    assert len(aligned) == 1
    assert aligned.loc[0, "s1"] == 5
    assert aligned.loc[0, "s2"] == 15


def test_compute_group_metrics_perfect():
    df = pd.DataFrame({
        "group": [0, 0, 1, 1],
        "compound_id": [1, 1, 2, 2],
    })
    metrics = compute_group_metrics(df)
    assert metrics["f1"] == 1.0
    assert metrics["ari"] == 1.0


def test_compute_group_metrics_imperfect():
    df = pd.DataFrame({
        "group": [0, 0, 1, 1],
        "compound_id": [1, 2, 1, 2],
    })
    metrics = compute_group_metrics(df)
    assert 0 <= metrics["precision"] < 1.0


def test_peak_table_from_ground_truth():
    gt = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "compound_id": [0, 1],
            "mz_apex": [100.0, 200.0],
            "rt_apex": [1.0, 2.0],
            "intensity": [10, 20],
            "mz_min": [99.9, 199.9],
            "mz_max": [100.1, 200.1],
            "rt_min": [0.9, 1.9],
            "rt_max": [1.1, 2.1],
        }
    )
    peaks = peak_table_from_ground_truth(gt)
    assert set(peaks.columns) == {"sample", "mz", "rt", "intensity"}
    assert len(peaks) == 2
    assert peaks.loc[1, "rt"] == 2.0


def test_run_pipeline(tmp_path):
    dataset = generate_synthetic_dataset(
        tmp_path,
        n_chemicals=1,
        n_samples_per_group=1,
        mzml_max_rt=20,
        top_n=1,
    )
    metrics = run_pipeline(
        dataset,
        out_dir=tmp_path,
        mz_tol=0.02,
        rt_tol=0.1,
    )
    assert set(metrics) == {"precision", "recall", "f1", "ari"}
    assert (tmp_path / "aligned.csv").is_file()
