from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vimms.AssignmentNoiseLabels import label_for
from vimms.AssignmentNoisePeaks import (
    _add_candidate_ions,
    _add_coeluting_isobars,
    _add_diffuse_background,
    _add_matched_decoy_clusters,
    _add_structured_interferents,
    _finalize_peak_table,
    _trim_peaks,
)

try:
    from vimms.Common import ADDUCT_TERMS, C13_MZ_DIFF, PROTON_MASS

    SODIUM_ADDUCT_MASS = float(ADDUCT_TERMS["M+Na"][1])
except ModuleNotFoundError:
    # Keep this picked-table generator usable from lightweight downstream
    # packages that only need assignment artifacts, not the full scan simulator.
    PROTON_MASS = 1.00727645199076
    C13_MZ_DIFF = 1.0033548378
    SODIUM_ADDUCT_MASS = 22.989218


ASSIGNMENT_ROLES: tuple[str, ...] = (
    "[M+H]+",
    "M+1",
    "M+2",
    "[M+Na]+",
    "fragment-H2O+H",
)

H2O_MASS = 18.010564684


@dataclass(frozen=True)
class AssignmentNoiseProfile:
    """Controls picked-peak assignment artifacts for labelled local problems.

    Retention time is exported in minutes because the IonClassifier feature
    contract expects local RT values on a roughly 0-15 minute scale. This module
    intentionally bypasses external peak pickers: it produces picked-like peak
    tables and truth labels directly from VIMMS-style chemical/adduct concepts.
    """

    n_candidates: int = 2
    min_peaks: int = 5
    max_peaks: int = 36
    n_bio_samples: int = 24
    n_qc_samples: int = 6
    n_blank_samples: int = 6
    mass_min: float = 120.0
    mass_max: float = 750.0
    rt_min: float = 1.2
    rt_max: float = 12.5
    sigma_ppm: float = 4.0
    mu_ppm: float = 0.0
    sigma_rt: float = 0.08
    lambda_clutter: float = 8.0
    p_present: float = 0.65
    p_missing: float = 0.12
    p_blank: float = 0.25
    p_blank_contaminant: float = 0.30
    p_interferent: float = 0.25
    p_near_rt_interferent: float = 0.40
    p_adversarial_interferent: float = 0.12
    p_coeluting_isobar: float = 0.22
    p_split: float = 0.08
    p_merge: float = 0.04
    p_false_ms2: float = 0.08
    p_matched_decoy: float = 0.35
    p_matched_decoy_role: float = 0.65
    p_decoy_false_ms2: float = 0.18
    p_in_source_fragment: float = 0.32
    evidence_factor: float = 1.0
    severe_noise_context: float = 0.0
    matched_decoy_context: float = 0.0


@dataclass(frozen=True)
class AssignmentScenarioConfig:
    """Configuration for one labelled assignment-noise local problem."""

    seed: int = 0
    profile: AssignmentNoiseProfile = field(default_factory=AssignmentNoiseProfile)
    roles: tuple[str, ...] = ASSIGNMENT_ROLES
    present_pattern: tuple[int, ...] | None = None


def theoretical_mz(neutral_mass: float, role: str) -> float:
    """Return the theoretical m/z for an assignment role."""

    if role == "[M+H]+":
        return float(neutral_mass) + PROTON_MASS
    if role == "M+1":
        return float(neutral_mass) + PROTON_MASS + C13_MZ_DIFF
    if role == "M+2":
        return float(neutral_mass) + PROTON_MASS + 2.0 * C13_MZ_DIFF
    if role == "[M+Na]+":
        return float(neutral_mass) + SODIUM_ADDUCT_MASS
    if role == "fragment-H2O+H":
        return float(neutral_mass) - H2O_MASS + PROTON_MASS
    raise ValueError(f"Unsupported assignment role: {role}")


def role_properties(role: str, carbon_count: float) -> dict[str, float]:
    """Return isotope/adduct priors used by the picked-table generator."""

    c13_1 = max(0.001, 0.011 * float(carbon_count))
    c13_2 = max(0.0005, (0.011 * float(carbon_count)) ** 2 / 2.0)
    if role == "[M+H]+":
        return {
            "isotope_index": 0,
            "expected_rel_intensity": 1.0,
            "adduct_prior": 0.95,
            "fragment_prior": 0.0,
            "detect_base": 0.92,
        }
    if role == "M+1":
        return {
            "isotope_index": 1,
            "expected_rel_intensity": c13_1,
            "adduct_prior": 0.75,
            "fragment_prior": 0.0,
            "detect_base": 0.78,
        }
    if role == "M+2":
        return {
            "isotope_index": 2,
            "expected_rel_intensity": c13_2,
            "adduct_prior": 0.45,
            "fragment_prior": 0.0,
            "detect_base": 0.42,
        }
    if role == "[M+Na]+":
        return {
            "isotope_index": 0,
            "expected_rel_intensity": 0.22,
            "adduct_prior": 0.55,
            "fragment_prior": 0.0,
            "detect_base": 0.58,
        }
    if role == "fragment-H2O+H":
        return {
            "isotope_index": 0,
            "expected_rel_intensity": 0.12,
            "adduct_prior": 0.25,
            "fragment_prior": 0.55,
            "detect_base": 0.38,
        }
    raise ValueError(f"Unsupported assignment role: {role}")


def _validate_config(config: AssignmentScenarioConfig) -> None:
    profile = config.profile
    if profile.n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    if profile.min_peaks < 0:
        raise ValueError("min_peaks must be non-negative")
    if profile.max_peaks < max(profile.min_peaks, 1):
        raise ValueError("max_peaks must be at least min_peaks and positive")
    if not config.roles:
        raise ValueError("roles must not be empty")
    if config.present_pattern is not None and len(config.present_pattern) != profile.n_candidates:
        raise ValueError("present_pattern length must match n_candidates")



def _presence_pattern(
    rng: np.random.Generator,
    profile: AssignmentNoiseProfile,
    present_pattern: tuple[int, ...] | None,
) -> np.ndarray:
    if present_pattern is not None:
        return np.asarray(present_pattern, dtype=np.int8)
    present = (rng.random(profile.n_candidates) < profile.p_present).astype(np.int8)
    if not present.any():
        present[0] = 1
    if profile.n_candidates > 1 and present.all() and profile.p_matched_decoy > 0.0:
        present[-1] = 0
    return present


def _make_candidate_table(
    rng: np.random.Generator,
    profile: AssignmentNoiseProfile,
    present: np.ndarray,
) -> pd.DataFrame:
    base_mass = float(rng.uniform(profile.mass_min, profile.mass_max))
    base_rt = float(rng.uniform(profile.rt_min, profile.rt_max))
    base_c = int(rng.integers(5, 45))
    rows: list[dict[str, Any]] = []
    for candidate_index in range(profile.n_candidates):
        if candidate_index == 0:
            neutral_mass = base_mass
            pred_rt = base_rt
            carbon_count = base_c
        elif candidate_index == 1:
            neutral_mass = max(50.0, base_mass + rng.normal(0.0, 0.018))
            pred_rt = float(np.clip(base_rt + rng.normal(0.0, 0.08), 0.3, 15.0))
            carbon_count = int(np.clip(base_c + rng.integers(-3, 4), 2, 70))
        else:
            neutral_mass = max(50.0, base_mass + rng.normal(0.0, rng.uniform(0.4, 4.0)))
            pred_rt = float(np.clip(base_rt + rng.normal(0.0, rng.uniform(0.25, 1.1)), 0.3, 15.0))
            carbon_count = int(np.clip(base_c + rng.integers(-8, 9), 2, 70))
        is_present = int(present[candidate_index])
        prior = float(rng.beta(5.0, 2.2) if is_present else rng.beta(2.2, 5.0))
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": f"vimms_cand_{candidate_index}",
                "compound_id": f"vimms_compound_{candidate_index}",
                "chemical_id": f"vimms_chemical_{candidate_index}",
                "neutral_mass": neutral_mass,
                "pred_rt": pred_rt,
                "rt_uncertainty": float(rng.uniform(0.05, 0.35)),
                "carbon_count": carbon_count,
                "hetero_count": int(rng.integers(2, 13)),
                "candidate_prior": prior,
                "expected_log_intensity": float(rng.normal(15.5, 1.1)),
                "compound_class": int(rng.integers(0, 3)),
                "present": is_present,
                "source_type": "target_candidate" if candidate_index == 0 else "candidate",
            }
        )
    return pd.DataFrame(rows)


def _make_ion_table(candidates: pd.DataFrame, roles: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        candidate_index = int(candidate["candidate_index"])
        for role_index, role in enumerate(roles):
            props = role_properties(role, float(candidate["carbon_count"]))
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": str(candidate["candidate_id"]),
                    "chemical_id": str(candidate["chemical_id"]),
                    "role_index": role_index,
                    "role": role,
                    "label": label_for(candidate_index, role_index, len(roles)),
                    "theoretical_mz": theoretical_mz(float(candidate["neutral_mass"]), role),
                    "expected_rt": float(candidate["pred_rt"]),
                    "rt_uncertainty": float(candidate["rt_uncertainty"]),
                    **props,
                }
            )
    return pd.DataFrame(rows)



def _make_ion_truth_table(
    peak_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    ion_table: pd.DataFrame,
) -> pd.DataFrame:
    observed = (
        peak_table.loc[peak_table["true_label"] > 0, ["true_label", "peak_id", "source_type"]]
        .drop_duplicates("true_label")
        .set_index("true_label")
        .to_dict("index")
    )
    rows: list[dict[str, Any]] = []
    present_lookup = candidate_table.set_index("candidate_index")["present"].astype(int).to_dict()
    for _, ion in ion_table.iterrows():
        candidate_index = int(ion["candidate_index"])
        present = int(present_lookup.get(candidate_index, 0))
        label = int(ion["label"])
        obs = observed.get(label)
        missing = bool(present and obs is None)
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": str(ion["candidate_id"]),
                "role_index": int(ion["role_index"]),
                "role": str(ion["role"]),
                "label": label,
                "candidate_present": present,
                "observed": int(obs is not None),
                "observed_peak_id": "" if obs is None else str(obs["peak_id"]),
                "observed_source_type": "" if obs is None else str(obs["source_type"]),
                "missing_reason": "missing_true_companion" if missing else "",
            }
        )
    return pd.DataFrame(rows)


def generate_assignment_peak_table(
    config: AssignmentScenarioConfig | None = None,
) -> dict[str, Any]:
    """Generate one VIMMS assignment-noise picked-peak table.

    Returns a dictionary with ``peak_table``, ``candidate_table``, ``ion_table``,
    ``truth_table`` (peak-level truth), ``ion_truth_table`` (including missing
    true companions), and ``scenario_metadata``. Spike noise is intentionally not
    used here because VIMMS spike peaks are scan-level additions and are not
    chemical-backed or fragmentable.
    """

    if config is None:
        config = AssignmentScenarioConfig()
    _validate_config(config)

    rng = np.random.default_rng(config.seed)
    profile = config.profile
    roles = tuple(config.roles)
    present = _presence_pattern(rng, profile, config.present_pattern)
    candidate_table = _make_candidate_table(rng, profile, present)
    ion_table = _make_ion_table(candidate_table, roles)

    latent_profiles: dict[int, np.ndarray] = {}
    for _, candidate in candidate_table.iterrows():
        candidate_index = int(candidate["candidate_index"])
        latent_profiles[candidate_index] = rng.normal(
            float(candidate["expected_log_intensity"]),
            rng.uniform(0.35, 0.95),
            size=profile.n_bio_samples,
        )

    peaks: list[dict[str, Any]] = []
    _add_candidate_ions(peaks, rng, candidate_table, ion_table, profile, roles, latent_profiles)
    _add_structured_interferents(peaks, rng, candidate_table, ion_table, profile)
    _add_coeluting_isobars(peaks, rng, candidate_table, ion_table, profile)
    _add_matched_decoy_clusters(peaks, rng, candidate_table, ion_table, profile, latent_profiles)
    _add_diffuse_background(
        peaks, rng, candidate_table, profile, int(rng.poisson(profile.lambda_clutter))
    )

    while len(peaks) < profile.min_peaks:
        _add_diffuse_background(peaks, rng, candidate_table, profile, 1)
    peaks = _trim_peaks(peaks, rng, profile.max_peaks)
    peak_table = _finalize_peak_table(peaks, rng)

    truth_columns = [
        "peak_id",
        "true_label",
        "true_label_text",
        "is_true_ion",
        "is_background",
        "source_type",
        "candidate_index",
        "candidate_id",
        "role",
        "parent_chemical_id",
        "decoy_candidate",
        "decoy_role",
        "artifact_parent_peak_id",
        "merged_from_label",
        "false_ms2_support",
    ]
    truth_table = peak_table[truth_columns].copy()
    ion_truth_table = _make_ion_truth_table(peak_table, candidate_table, ion_table)
    source_counts = peak_table["source_type"].value_counts().sort_index().to_dict()
    missing_count = int((ion_truth_table["missing_reason"] == "missing_true_companion").sum())
    if missing_count:
        source_counts["missing_true_companion"] = missing_count

    scenario_metadata = {
        "seed": int(config.seed),
        "roles": list(roles),
        "rt_unit": "minutes",
        "profile": asdict(profile),
        "n_peaks": int(len(peak_table)),
        "n_candidates": int(len(candidate_table)),
        "n_true_ion_peaks": int((peak_table["true_label"] > 0).sum()),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "run_context": {
            "sigma_ppm": float(profile.sigma_ppm),
            "mu_ppm": float(profile.mu_ppm),
            "sigma_rt": float(profile.sigma_rt),
            "lambda_clutter": float(profile.lambda_clutter),
            "p_missing": float(profile.p_missing),
            "p_interferent": float(profile.p_interferent),
            "p_split": float(profile.p_split),
            "p_blank": float(profile.p_blank),
            "p_ms2_fp": float(profile.p_false_ms2),
            "severe_noise_context": float(profile.severe_noise_context),
            "matched_decoy_context": float(profile.matched_decoy_context),
        },
        "notes": [
            "Picked-like peak table generated directly from assignment artifacts.",
            "UniformSpikeNoise is not used because VIMMS spike peaks are not chemical-backed.",
        ],
    }

    return {
        "peak_table": peak_table,
        "candidate_table": candidate_table,
        "ion_table": ion_table,
        "truth_table": truth_table,
        "ion_truth_table": ion_truth_table,
        "scenario_metadata": scenario_metadata,
    }


def write_assignment_peak_table(
    artifact: Mapping[str, Any],
    output_dir: str | Path,
    prefix: str = "assignment",
) -> dict[str, Path]:
    """Write an assignment-noise artifact bundle to CSV/JSON files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "peak_table": output_path / f"{prefix}_peaks.csv",
        "candidate_table": output_path / f"{prefix}_candidates.csv",
        "ion_table": output_path / f"{prefix}_ions.csv",
        "truth_table": output_path / f"{prefix}_truth.csv",
        "ion_truth_table": output_path / f"{prefix}_ion_truth.csv",
        "scenario_metadata": output_path / f"{prefix}_metadata.json",
    }
    for key in ("peak_table", "candidate_table", "ion_table", "truth_table", "ion_truth_table"):
        table = artifact.get(key)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"artifact[{key!r}] must be a pandas DataFrame")
        table.to_csv(paths[key], index=False)
    paths["scenario_metadata"].write_text(
        json.dumps(artifact.get("scenario_metadata", {}), indent=2),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "ASSIGNMENT_ROLES",
    "AssignmentNoiseProfile",
    "AssignmentScenarioConfig",
    "generate_assignment_peak_table",
    "label_for",
    "role_properties",
    "theoretical_mz",
    "write_assignment_peak_table",
]
