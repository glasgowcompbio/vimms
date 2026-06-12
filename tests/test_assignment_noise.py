from __future__ import annotations

import json

from vimms.AssignmentNoise import (
    AssignmentNoiseProfile,
    AssignmentScenarioConfig,
    generate_assignment_peak_table,
    label_for,
    write_assignment_peak_table,
)


def _stress_profile(**overrides):
    values = {
        "n_candidates": 3,
        "min_peaks": 10,
        "max_peaks": 200,
        "lambda_clutter": 8.0,
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


def test_assignment_peak_table_contains_hard_artifacts_and_labels() -> None:
    artifact = generate_assignment_peak_table(
        AssignmentScenarioConfig(
            seed=17,
            profile=_stress_profile(),
            present_pattern=(1, 0, 1),
        )
    )

    peaks = artifact["peak_table"]
    candidates = artifact["candidate_table"]
    ions = artifact["ion_table"]
    truth = artifact["truth_table"]
    ion_truth = artifact["ion_truth_table"]
    metadata = artifact["scenario_metadata"]

    required_peak_columns = {
        "peak_id",
        "mz",
        "rt",
        "intensity",
        "log_intensity",
        "peak_width",
        "quality",
        "missingness",
        "qc_cv",
        "blank_ratio",
        "blank_frequency",
        "profile_mean",
        "profile_variance",
        "profile_skew",
        "local_density",
        "max_profile_corr",
        "ms2_available",
        "best_ms2_score",
        "ms2_score_cand_0",
        "ms2_score_cand_1",
        "ms2_score_cand_2",
        "source_type",
        "candidate_id",
        "candidate_index",
        "role",
        "is_true_ion",
        "is_background",
        "parent_chemical_id",
        "true_label",
        "false_ms2_support",
    }
    assert required_peak_columns.issubset(peaks.columns)
    assert {"candidate_index", "present", "neutral_mass", "pred_rt"}.issubset(candidates.columns)
    assert {"candidate_index", "role", "label", "theoretical_mz"}.issubset(ions.columns)
    assert len(truth) == len(peaks)
    assert len(ion_truth) == len(ions)

    source_types = set(peaks["source_type"])
    assert {
        "candidate_ion",
        "in_source_fragment",
        "split_or_shoulder",
        "merged_peak",
        "near_rt_structured_interferent",
        "coeluting_isobar",
        "matched_decoy_cluster",
        "blank_contaminant",
    }.issubset(source_types)

    true_peaks = peaks[peaks["true_label"] > 0]
    assert not true_peaks.empty
    assert (true_peaks["is_true_ion"] == 1).all()
    assert (peaks.loc[peaks["true_label"] == 0, "is_background"] == 1).all()

    first_true = true_peaks.iloc[0]
    role_index = list(artifact["scenario_metadata"]["roles"]).index(first_true["role"])
    assert int(first_true["true_label"]) == label_for(
        int(first_true["candidate_index"]),
        role_index,
        len(artifact["scenario_metadata"]["roles"]),
    )

    background = peaks[peaks["true_label"] == 0]
    assert (background["false_ms2_support"] == 1).any()
    assert metadata["rt_unit"] == "minutes"
    assert metadata["source_counts"]["matched_decoy_cluster"] > 0
    assert "UniformSpikeNoise is not used" in " ".join(metadata["notes"])


def test_assignment_peak_table_records_missing_true_companions() -> None:
    artifact = generate_assignment_peak_table(
        AssignmentScenarioConfig(
            seed=29,
            profile=_stress_profile(
                p_missing=1.0,
                p_in_source_fragment=0.0,
                p_split=0.0,
                p_merge=0.0,
                p_coeluting_isobar=0.0,
                p_matched_decoy=0.0,
                p_interferent=0.0,
            ),
            present_pattern=(1, 0, 0),
        )
    )

    ion_truth = artifact["ion_truth_table"]
    missing = ion_truth[ion_truth["missing_reason"] == "missing_true_companion"]
    assert not missing.empty
    assert (missing["candidate_present"] == 1).all()
    assert artifact["scenario_metadata"]["source_counts"]["missing_true_companion"] == len(missing)


def test_assignment_peak_table_writer(tmp_path) -> None:
    artifact = generate_assignment_peak_table(
        AssignmentScenarioConfig(
            seed=3, profile=_stress_profile(max_peaks=120), present_pattern=(1, 0, 1)
        )
    )
    paths = write_assignment_peak_table(artifact, tmp_path, prefix="demo")

    assert set(paths) == {
        "peak_table",
        "candidate_table",
        "ion_table",
        "truth_table",
        "ion_truth_table",
        "scenario_metadata",
    }
    for path in paths.values():
        assert path.exists()
    metadata = json.loads(paths["scenario_metadata"].read_text(encoding="utf-8"))
    assert metadata["n_peaks"] == len(artifact["peak_table"])
