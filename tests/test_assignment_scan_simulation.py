from __future__ import annotations

import json

from vimms.AssignmentNoise import AssignmentNoiseProfile
from vimms.AssignmentScanSimulation import (
    AssignmentScanSimulationConfig,
    generate_assignment_scan_artifact,
    write_assignment_scan_artifact,
)


def _scan_stress_profile(**overrides):
    values = {
        "n_candidates": 3,
        "min_peaks": 10,
        "max_peaks": 90,
        "lambda_clutter": 6.0,
        "sigma_ppm": 5.0,
        "sigma_rt": 0.10,
        "p_missing": 0.05,
        "p_blank": 1.0,
        "p_blank_contaminant": 1.0,
        "p_interferent": 1.0,
        "p_near_rt_interferent": 1.0,
        "p_adversarial_interferent": 1.0,
        "p_coeluting_isobar": 1.0,
        "p_split": 1.0,
        "p_merge": 1.0,
        "p_false_ms2": 1.0,
        "p_matched_decoy": 1.0,
        "p_matched_decoy_role": 1.0,
        "p_decoy_false_ms2": 1.0,
        "p_in_source_fragment": 1.0,
        "severe_noise_context": 1.0,
        "matched_decoy_context": 1.0,
    }
    values.update(overrides)
    return AssignmentNoiseProfile(**values)


def test_assignment_scan_artifact_writes_mzml_and_scan_tables(tmp_path) -> None:
    artifact = generate_assignment_scan_artifact(
        AssignmentScanSimulationConfig(
            seed=17,
            profile=_scan_stress_profile(),
            present_pattern=(1, 0, 1),
            output_dir=tmp_path,
            prefix="scan_demo",
            topn_n=12,
        )
    )

    mzml_path = artifact["mzml_path"]
    assert mzml_path is not None
    assert mzml_path.exists()
    assert artifact["scenario_metadata"]["source_mode"] == "scan"
    assert artifact["scenario_metadata"]["scan_simulation"]["spike_noise_used"] is False

    peaks = artifact["peak_table"]
    scans = artifact["scan_summary"]
    chemical_truth = artifact["chemical_truth_table"]
    assert not peaks.empty
    assert not scans.empty
    assert (scans["ms_level"] == 1).any()
    assert {"scan_backed", "scan_ms1_match_count", "rt_seconds", "ms2_scan_count"}.issubset(
        peaks.columns
    )
    assert peaks["scan_backed"].sum() > 0
    assert len(chemical_truth) == len(peaks)
    assert (chemical_truth["chemical_backed"] == 1).all()
    assert (chemical_truth["spike_noise_used"] == 0).all()


def test_assignment_scan_artifact_keeps_hard_artifacts_chemical_backed(tmp_path) -> None:
    artifact = generate_assignment_scan_artifact(
        AssignmentScanSimulationConfig(
            seed=19,
            profile=_scan_stress_profile(max_peaks=120),
            present_pattern=(1, 0, 1),
            output_dir=tmp_path,
            prefix="hard_demo",
            topn_n=16,
        )
    )
    chemical_truth = artifact["chemical_truth_table"]
    source_types = set(chemical_truth["source_type"])

    assert {
        "candidate_ion",
        "matched_decoy_cluster",
        "coeluting_isobar",
        "near_rt_structured_interferent",
        "blank_contaminant",
    }.issubset(source_types)
    assert (
        chemical_truth.loc[
            chemical_truth["source_type"].isin(
                ["matched_decoy_cluster", "coeluting_isobar", "near_rt_structured_interferent"]
            ),
            "chemical_backed",
        ]
        .eq(1)
        .all()
    )

    paths = write_assignment_scan_artifact(artifact, tmp_path / "bundle", prefix="scan")
    assert paths["peak_table"].exists()
    assert paths["chemical_truth_table"].exists()
    assert paths["scan_summary"].exists()
    metadata = json.loads(paths["scenario_metadata"].read_text(encoding="utf-8"))
    assert "internal truth" in " ".join(metadata["notes"])
