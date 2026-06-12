from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vimms.Common import ADDUCT_TERMS, C13_MZ_DIFF, PROTON_MASS


ASSIGNMENT_ROLES: tuple[str, ...] = (
    "[M+H]+",
    "M+1",
    "M+2",
    "[M+Na]+",
    "fragment-H2O+H",
)

H2O_MASS = 18.010564684
SODIUM_ADDUCT_MASS = float(ADDUCT_TERMS["M+Na"][1])


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


def label_for(candidate_index: int, role_index: int, n_roles: int) -> int:
    """Map candidate/role coordinates to a peak-assignment class label."""

    return 1 + int(candidate_index) * int(n_roles) + int(role_index)


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


def _skew(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    if sd < 1e-8:
        return 0.0
    return float(np.mean(((arr - arr.mean()) / sd) ** 3))


def _beta_score(rng: np.random.Generator, true: bool, fp_prob: float) -> float:
    if true:
        return float(rng.beta(9.0, 2.2))
    if rng.random() < fp_prob:
        return float(rng.beta(5.0, 2.4))
    return float(rng.beta(1.2, 8.0))


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


def _peak_row(
    *,
    rng: np.random.Generator,
    mz: float,
    rt: float,
    log_intensity: float,
    blank_ratio: float,
    blank_frequency: float,
    qc_cv: float,
    quality: float,
    peak_width: float,
    missingness: float,
    profile: np.ndarray,
    source_type: str,
    true_label: int,
    true_label_text: str,
    n_candidates: int,
    true_candidate: int | None = None,
    candidate_id: str = "",
    role: str = "",
    parent_chemical_id: str = "",
    fp_prob: float = 0.03,
    ms2_true_score: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mz": float(mz),
        "rt": float(rt),
        "intensity": float(np.exp(log_intensity)),
        "log_intensity": float(log_intensity),
        "peak_width": float(peak_width),
        "quality": float(np.clip(quality, 0.0, 1.0)),
        "missingness": float(np.clip(missingness, 0.0, 1.0)),
        "qc_cv": float(max(0.0, qc_cv)),
        "blank_ratio": float(max(0.0, blank_ratio)),
        "blank_frequency": float(np.clip(blank_frequency, 0.0, 1.0)),
        "profile_mean": float(np.mean(profile)),
        "profile_variance": float(np.var(profile)),
        "profile_skew": _skew(profile),
        "local_density": 0.0,
        "max_profile_corr": 0.0,
        "source_type": source_type,
        "candidate_index": -1 if true_candidate is None else int(true_candidate),
        "candidate_id": candidate_id,
        "role": role,
        "is_true_ion": int(true_label > 0),
        "is_background": int(true_label == 0),
        "parent_chemical_id": parent_chemical_id,
        "true_label": int(true_label),
        "true_label_text": true_label_text,
        "artifact_parent_key": "",
        "artifact_parent_peak_id": "",
        "merged_from_label": 0,
        "decoy_candidate": -1,
        "decoy_role": "",
        "ion_key": "",
        "_profile": profile.astype(np.float32),
    }

    ms2_scores = []
    for candidate_index in range(n_candidates):
        is_true = bool(ms2_true_score and true_candidate == candidate_index)
        score = _beta_score(rng, is_true, fp_prob)
        row[f"ms2_score_cand_{candidate_index}"] = score
        ms2_scores.append(score)

    available_prob = 0.70 if ms2_true_score else min(0.85, 0.25 + 0.55 * fp_prob)
    available = bool(rng.random() < available_prob)
    if not available:
        for candidate_index in range(n_candidates):
            row[f"ms2_score_cand_{candidate_index}"] *= float(rng.uniform(0.0, 0.25))
        ms2_scores = [
            row[f"ms2_score_cand_{candidate_index}"] for candidate_index in range(n_candidates)
        ]
    row["ms2_available"] = float(available)
    row["best_ms2_score"] = float(max(ms2_scores))
    row["false_ms2_support"] = int(true_label == 0 and row["best_ms2_score"] >= 0.45)
    if metadata:
        row.update(dict(metadata))
    return row


def _finalize_peak_table(peaks: list[dict[str, Any]], rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(peaks)
    if df.empty:
        return df
    mz = df["mz"].to_numpy(float)
    rt = df["rt"].to_numpy(float)
    local_density = []
    for i in range(len(df)):
        dmz = np.abs(mz - mz[i])
        drt = np.abs(rt - rt[i])
        local_density.append(int(((dmz <= 0.05) & (drt <= 0.25)).sum() - 1))

    profiles = np.stack(df["_profile"].to_numpy())
    if len(df) > 1:
        corr = np.corrcoef(profiles)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        max_corr = np.max(corr, axis=1)
    else:
        max_corr = np.zeros(len(df))

    df["local_density"] = np.asarray(local_density, dtype=float)
    df["max_profile_corr"] = max_corr
    df = df.drop(columns=["_profile"])
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    df["peak_id"] = [f"vimms_peak_{i}" for i in range(len(df))]

    parent_lookup = (
        df.loc[(df["true_label"] > 0) & (df["ion_key"] != ""), ["ion_key", "peak_id"]]
        .drop_duplicates("ion_key")
        .set_index("ion_key")["peak_id"]
        .to_dict()
    )
    if "artifact_parent_key" in df.columns:
        df["artifact_parent_peak_id"] = df["artifact_parent_key"].map(parent_lookup).fillna("")
    return df


def _add_diffuse_background(
    peaks: list[dict[str, Any]],
    rng: np.random.Generator,
    candidates: pd.DataFrame,
    profile: AssignmentNoiseProfile,
    n: int,
) -> None:
    mz_min = float(candidates["neutral_mass"].min() - 25.0)
    mz_max = float(candidates["neutral_mass"].max() + 45.0)
    rt_center = float(candidates["pred_rt"].mean())
    for _ in range(n):
        log_i = float(rng.normal(14.2, 1.5))
        peak_profile = rng.normal(log_i, rng.uniform(0.4, 1.5), size=profile.n_bio_samples)
        is_blank = rng.random() < max(profile.p_blank, profile.p_blank_contaminant)
        peaks.append(
            _peak_row(
                rng=rng,
                mz=float(rng.uniform(mz_min, mz_max)),
                rt=float(np.clip(rng.normal(rt_center, 1.4), 0.1, 15.0)),
                log_intensity=log_i,
                blank_ratio=float(
                    rng.lognormal(1.2, 0.8) if is_blank else rng.lognormal(-2.1, 0.8)
                ),
                blank_frequency=float(
                    rng.uniform(0.4, 1.0) if is_blank else rng.uniform(0.0, 0.3)
                ),
                qc_cv=float(rng.uniform(0.18, 0.85)),
                quality=float(rng.beta(3.0, 3.0)),
                peak_width=float(rng.uniform(0.03, 0.45)),
                missingness=float(rng.uniform(0.05, 0.75)),
                profile=peak_profile,
                source_type="blank_contaminant" if is_blank else "diffuse_background",
                true_label=0,
                true_label_text="background/noise/artifact",
                n_candidates=profile.n_candidates,
                fp_prob=profile.p_false_ms2,
            )
        )


def _add_candidate_ions(
    peaks: list[dict[str, Any]],
    rng: np.random.Generator,
    candidates: pd.DataFrame,
    ions: pd.DataFrame,
    profile: AssignmentNoiseProfile,
    roles: tuple[str, ...],
    latent_profiles: dict[int, np.ndarray],
) -> None:
    for _, candidate in candidates.iterrows():
        candidate_index = int(candidate["candidate_index"])
        if int(candidate["present"]) <= 0:
            continue
        cand_ions = ions[ions["candidate_index"] == candidate_index].sort_values("role_index")
        log_a = float(candidate["expected_log_intensity"])
        abundance_factor = 1.0 / (1.0 + np.exp(-(log_a - 14.5)))
        emitted_any = False
        for _, ion in cand_ions.iterrows():
            role = str(ion["role"])
            role_index = int(ion["role_index"])
            detect_prob = (
                float(ion["detect_base"])
                * (0.55 + 0.55 * abundance_factor)
                * profile.evidence_factor
                * (1.0 - 0.55 * profile.p_missing)
            )
            if role == "fragment-H2O+H":
                detect_prob = profile.p_in_source_fragment * (0.45 + 0.55 * abundance_factor)
                if profile.p_in_source_fragment >= 1.0:
                    detect_prob = 1.0
            detect_prob = float(np.clip(detect_prob, 0.02, 0.99))
            detected = bool(rng.random() < detect_prob)
            if role == "[M+H]+" and not emitted_any:
                detected = detected or bool(rng.random() < 0.80)
            if not detected:
                continue

            theory_mz = float(ion["theoretical_mz"])
            ppm_error = profile.mu_ppm + rng.normal(0.0, profile.sigma_ppm)
            observed_mz = theory_mz * (1.0 + ppm_error * 1e-6)
            observed_rt = float(
                np.clip(
                    float(candidate["pred_rt"]) + rng.normal(0.0, profile.sigma_rt), 0.05, 15.0
                )
            )
            rel_i = max(float(ion["expected_rel_intensity"]), 1e-6) * float(
                rng.lognormal(0.0, 0.40)
            )
            log_i = float(log_a + np.log(rel_i) + rng.normal(0.0, 0.35))
            peak_profile = (
                latent_profiles[candidate_index]
                + np.log(max(rel_i, 1e-6))
                + rng.normal(0.0, rng.uniform(0.08, 0.28), size=profile.n_bio_samples)
            )
            label = label_for(candidate_index, role_index, len(roles))
            ion_key = f"{candidate_index}:{role}"
            source_type = "in_source_fragment" if role == "fragment-H2O+H" else "candidate_ion"
            peaks.append(
                _peak_row(
                    rng=rng,
                    mz=observed_mz,
                    rt=observed_rt,
                    log_intensity=log_i,
                    blank_ratio=float(rng.lognormal(-3.0, 0.5)),
                    blank_frequency=float(rng.uniform(0.0, 0.12)),
                    qc_cv=float(rng.uniform(0.03, 0.18)),
                    quality=float(rng.beta(8.0, 1.8)),
                    peak_width=float(rng.uniform(0.04, 0.18)),
                    missingness=float(
                        np.clip(profile.p_missing + rng.normal(0.0, 0.08), 0.0, 0.75)
                    ),
                    profile=peak_profile,
                    source_type=source_type,
                    true_label=label,
                    true_label_text=f"{candidate['candidate_id']}:{role}",
                    true_candidate=candidate_index,
                    candidate_id=str(candidate["candidate_id"]),
                    role=role,
                    parent_chemical_id=str(candidate["chemical_id"]),
                    n_candidates=profile.n_candidates,
                    fp_prob=profile.p_false_ms2,
                    ms2_true_score=role in {"[M+H]+", "[M+Na]+", "fragment-H2O+H"},
                    metadata={"ion_key": ion_key},
                )
            )
            emitted_any = True

            if rng.random() < profile.p_split:
                split_profile = peak_profile + rng.normal(0.0, 0.15, size=profile.n_bio_samples)
                peaks.append(
                    _peak_row(
                        rng=rng,
                        mz=observed_mz * (1.0 + rng.normal(0.0, 1.2) * 1e-6),
                        rt=float(np.clip(observed_rt + rng.normal(0.0, 0.018), 0.05, 15.0)),
                        log_intensity=float(log_i + rng.normal(-1.2, 0.4)),
                        blank_ratio=float(rng.lognormal(-2.6, 0.8)),
                        blank_frequency=float(rng.uniform(0.0, 0.18)),
                        qc_cv=float(rng.uniform(0.15, 0.55)),
                        quality=float(rng.beta(2.2, 4.5)),
                        peak_width=float(rng.uniform(0.02, 0.08)),
                        missingness=float(rng.uniform(0.1, 0.65)),
                        profile=split_profile,
                        source_type="split_or_shoulder",
                        true_label=0,
                        true_label_text="background/noise/artifact",
                        parent_chemical_id=str(candidate["chemical_id"]),
                        n_candidates=profile.n_candidates,
                        fp_prob=profile.p_false_ms2,
                        metadata={
                            "artifact_parent_key": ion_key,
                            "merged_from_label": int(label),
                        },
                    )
                )

            if rng.random() < profile.p_merge:
                merged_profile = peak_profile + rng.normal(0.0, 0.18, size=profile.n_bio_samples)
                peaks.append(
                    _peak_row(
                        rng=rng,
                        mz=observed_mz * (1.0 + rng.normal(0.0, 2.0) * 1e-6),
                        rt=float(np.clip(observed_rt + rng.normal(0.0, 0.035), 0.05, 15.0)),
                        log_intensity=float(log_i + rng.normal(0.15, 0.25)),
                        blank_ratio=float(rng.lognormal(-1.7, 0.8)),
                        blank_frequency=float(rng.uniform(0.0, 0.28)),
                        qc_cv=float(rng.uniform(0.12, 0.42)),
                        quality=float(rng.beta(4.5, 2.4)),
                        peak_width=float(rng.uniform(0.14, 0.42)),
                        missingness=float(rng.uniform(0.02, 0.45)),
                        profile=merged_profile,
                        source_type="merged_peak",
                        true_label=0,
                        true_label_text="background/noise/artifact",
                        parent_chemical_id=str(candidate["chemical_id"]),
                        n_candidates=profile.n_candidates,
                        fp_prob=profile.p_false_ms2,
                        metadata={
                            "artifact_parent_key": ion_key,
                            "merged_from_label": int(label),
                        },
                    )
                )


def _add_structured_interferents(
    peaks: list[dict[str, Any]],
    rng: np.random.Generator,
    candidates: pd.DataFrame,
    ions: pd.DataFrame,
    profile: AssignmentNoiseProfile,
) -> None:
    for _, ion in ions.iterrows():
        if rng.random() >= profile.p_interferent:
            continue
        candidate_index = int(ion["candidate_index"])
        candidate = candidates.loc[candidates["candidate_index"] == candidate_index].iloc[0]
        ppm_error = profile.mu_ppm + rng.normal(0.0, max(1.5, profile.sigma_ppm * 0.8))
        mz = float(ion["theoretical_mz"]) * (1.0 + ppm_error * 1e-6)
        near_rt = bool(rng.random() < profile.p_near_rt_interferent)
        if near_rt:
            rt_offset = rng.normal(0.0, max(0.04, 1.2 * profile.sigma_rt))
        else:
            rt_offset = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 1.8)
        rt = float(np.clip(float(candidate["pred_rt"]) + rt_offset, 0.05, 15.0))
        adversarial = bool(rng.random() < profile.p_adversarial_interferent)
        log_i = float(
            rng.normal(
                float(candidate["expected_log_intensity"]) - (0.05 if adversarial else 0.4),
                0.75 if adversarial else 0.9,
            )
        )
        peak_profile = rng.normal(
            log_i,
            rng.uniform(0.25 if adversarial else 0.35, 1.10 if adversarial else 1.25),
            size=profile.n_bio_samples,
        )
        is_blank = rng.random() < max(profile.p_blank, 0.15)
        peaks.append(
            _peak_row(
                rng=rng,
                mz=mz,
                rt=rt,
                log_intensity=log_i,
                blank_ratio=float(
                    rng.lognormal(0.45, 0.75)
                    if (is_blank and adversarial)
                    else (rng.lognormal(1.1, 0.7) if is_blank else rng.lognormal(-1.6, 0.7))
                ),
                blank_frequency=float(
                    rng.uniform(0.25, 0.9)
                    if (is_blank and adversarial)
                    else (rng.uniform(0.5, 1.0) if is_blank else rng.uniform(0.0, 0.3))
                ),
                qc_cv=float(rng.uniform(0.08, 0.45) if adversarial else rng.uniform(0.18, 0.75)),
                quality=float(rng.beta(6.0, 2.2) if adversarial else rng.beta(3.0, 3.2)),
                peak_width=float(
                    rng.uniform(0.04, 0.28) if adversarial else rng.uniform(0.04, 0.38)
                ),
                missingness=float(
                    rng.uniform(0.03, 0.48) if adversarial else rng.uniform(0.06, 0.68)
                ),
                profile=peak_profile,
                source_type=(
                    "near_rt_structured_interferent" if near_rt else "structured_interferent"
                ),
                true_label=0,
                true_label_text="background/noise/artifact",
                n_candidates=profile.n_candidates,
                fp_prob=profile.p_false_ms2,
                metadata={
                    "decoy_candidate": candidate_index,
                    "decoy_role": str(ion["role"]),
                },
            )
        )


def _add_coeluting_isobars(
    peaks: list[dict[str, Any]],
    rng: np.random.Generator,
    candidates: pd.DataFrame,
    ions: pd.DataFrame,
    profile: AssignmentNoiseProfile,
) -> None:
    for _, ion in ions.iterrows():
        if rng.random() >= profile.p_coeluting_isobar:
            continue
        candidate = candidates.loc[
            candidates["candidate_index"] == int(ion["candidate_index"])
        ].iloc[0]
        mz = float(ion["theoretical_mz"]) * (
            1.0 + rng.normal(0.0, max(0.6, profile.sigma_ppm * 0.4)) * 1e-6
        )
        rt = float(
            np.clip(
                float(candidate["pred_rt"]) + rng.normal(0.0, max(0.015, profile.sigma_rt * 0.35)),
                0.05,
                15.0,
            )
        )
        log_i = float(candidate["expected_log_intensity"] + rng.normal(-0.1, 0.55))
        peak_profile = rng.normal(log_i, rng.uniform(0.18, 0.55), size=profile.n_bio_samples)
        peaks.append(
            _peak_row(
                rng=rng,
                mz=mz,
                rt=rt,
                log_intensity=log_i,
                blank_ratio=float(rng.lognormal(-2.1, 0.6)),
                blank_frequency=float(rng.uniform(0.0, 0.22)),
                qc_cv=float(rng.uniform(0.04, 0.22)),
                quality=float(rng.beta(7.0, 2.0)),
                peak_width=float(rng.uniform(0.04, 0.20)),
                missingness=float(rng.uniform(0.02, 0.30)),
                profile=peak_profile,
                source_type="coeluting_isobar",
                true_label=0,
                true_label_text="background/noise/artifact",
                n_candidates=profile.n_candidates,
                fp_prob=profile.p_false_ms2,
                metadata={
                    "decoy_candidate": int(ion["candidate_index"]),
                    "decoy_role": str(ion["role"]),
                },
            )
        )


def _add_matched_decoy_clusters(
    peaks: list[dict[str, Any]],
    rng: np.random.Generator,
    candidates: pd.DataFrame,
    ions: pd.DataFrame,
    profile: AssignmentNoiseProfile,
    latent_profiles: dict[int, np.ndarray],
) -> None:
    candidate_indices = list(candidates["candidate_index"].astype(int))
    absent = list(
        candidates.loc[candidates["present"].astype(int) == 0, "candidate_index"].astype(int)
    )
    target_candidates = absent if absent else [int(rng.choice(candidate_indices))]
    role_priority = {
        "[M+H]+": 1.0,
        "M+1": 0.82,
        "M+2": 0.45,
        "[M+Na]+": 0.58,
        "fragment-H2O+H": 0.38,
    }
    for candidate_index in target_candidates:
        if rng.random() >= profile.p_matched_decoy:
            continue
        candidate = candidates.loc[candidates["candidate_index"] == candidate_index].iloc[0]
        cand_ions = ions[ions["candidate_index"] == candidate_index].sort_values("role_index")
        decoy_log_a = float(candidate["expected_log_intensity"] + rng.normal(-0.05, 0.55))
        if latent_profiles and rng.random() < 0.65:
            base_profile = latent_profiles[int(rng.choice(list(latent_profiles.keys())))].copy()
            base_profile = base_profile - base_profile.mean() + decoy_log_a
        else:
            base_profile = rng.normal(
                decoy_log_a, rng.uniform(0.35, 0.90), size=profile.n_bio_samples
            )
        cluster_rt = float(
            np.clip(
                float(candidate["pred_rt"]) + rng.normal(0.0, max(0.018, 0.45 * profile.sigma_rt)),
                0.05,
                15.0,
            )
        )
        emitted = 0
        for _, ion in cand_ions.iterrows():
            role = str(ion["role"])
            p_role = profile.p_matched_decoy_role * role_priority.get(role, 0.5)
            if role == "[M+H]+":
                p_role = max(p_role, 0.92)
            if rng.random() >= p_role:
                continue
            rel_i = max(float(ion["expected_rel_intensity"]), 1e-6) * float(
                rng.lognormal(0.0, 0.45)
            )
            peak_profile = (
                base_profile
                + np.log(max(rel_i, 1e-6))
                + rng.normal(0.0, rng.uniform(0.10, 0.32), size=profile.n_bio_samples)
            )
            row = _peak_row(
                rng=rng,
                mz=float(ion["theoretical_mz"])
                * (
                    1.0
                    + (profile.mu_ppm + rng.normal(0.0, max(1.2, 0.55 * profile.sigma_ppm))) * 1e-6
                ),
                rt=float(
                    np.clip(
                        cluster_rt + rng.normal(0.0, max(0.010, 0.35 * profile.sigma_rt)),
                        0.05,
                        15.0,
                    )
                ),
                log_intensity=float(decoy_log_a + np.log(rel_i) + rng.normal(0.0, 0.35)),
                blank_ratio=float(rng.lognormal(-2.25, 0.65)),
                blank_frequency=float(rng.uniform(0.0, 0.22)),
                qc_cv=float(rng.uniform(0.04, 0.24)),
                quality=float(rng.beta(7.0, 2.0)),
                peak_width=float(rng.uniform(0.04, 0.20)),
                missingness=float(rng.uniform(0.02, 0.32)),
                profile=peak_profile,
                source_type="matched_decoy_cluster",
                true_label=0,
                true_label_text="background/noise/artifact",
                n_candidates=profile.n_candidates,
                fp_prob=profile.p_false_ms2,
                metadata={
                    "decoy_candidate": int(candidate_index),
                    "decoy_role": role,
                },
            )
            if rng.random() < profile.p_decoy_false_ms2:
                row[f"ms2_score_cand_{candidate_index}"] = float(
                    max(row.get(f"ms2_score_cand_{candidate_index}", 0.0), rng.beta(5.0, 2.0))
                )
                row["ms2_available"] = 1.0
                row["best_ms2_score"] = float(
                    max(row["best_ms2_score"], row[f"ms2_score_cand_{candidate_index}"])
                )
                row["false_ms2_support"] = int(row["best_ms2_score"] >= 0.45)
            peaks.append(row)
            emitted += 1

        if emitted == 0:
            ion = cand_ions.iloc[0]
            peaks.append(
                _peak_row(
                    rng=rng,
                    mz=float(ion["theoretical_mz"])
                    * (
                        1.0
                        + (profile.mu_ppm + rng.normal(0.0, max(1.0, 0.45 * profile.sigma_ppm)))
                        * 1e-6
                    ),
                    rt=cluster_rt,
                    log_intensity=float(decoy_log_a + rng.normal(0.0, 0.35)),
                    blank_ratio=float(rng.lognormal(-2.2, 0.65)),
                    blank_frequency=float(rng.uniform(0.0, 0.20)),
                    qc_cv=float(rng.uniform(0.04, 0.22)),
                    quality=float(rng.beta(7.0, 2.0)),
                    peak_width=float(rng.uniform(0.04, 0.20)),
                    missingness=float(rng.uniform(0.02, 0.30)),
                    profile=base_profile + rng.normal(0.0, 0.18, size=profile.n_bio_samples),
                    source_type="matched_decoy_cluster",
                    true_label=0,
                    true_label_text="background/noise/artifact",
                    n_candidates=profile.n_candidates,
                    fp_prob=profile.p_false_ms2,
                    metadata={
                        "decoy_candidate": int(candidate_index),
                        "decoy_role": str(ion["role"]),
                    },
                )
            )


def _trim_peaks(
    peaks: list[dict[str, Any]],
    rng: np.random.Generator,
    max_peaks: int,
) -> list[dict[str, Any]]:
    if len(peaks) <= max_peaks:
        return peaks
    true_rows = [p for p in peaks if int(p["true_label"]) > 0]
    hard_types = {
        "matched_decoy_cluster",
        "coeluting_isobar",
        "near_rt_structured_interferent",
        "structured_interferent",
        "split_or_shoulder",
        "merged_peak",
    }
    hard_rows = [p for p in peaks if int(p["true_label"]) == 0 and p["source_type"] in hard_types]
    easy_rows = [
        p for p in peaks if int(p["true_label"]) == 0 and p["source_type"] not in hard_types
    ]
    remaining = max(0, max_peaks - len(true_rows))
    if len(hard_rows) > remaining:
        idx = rng.choice(len(hard_rows), size=remaining, replace=False)
        hard_rows = [hard_rows[int(i)] for i in idx]
        easy_rows = []
    else:
        remaining -= len(hard_rows)
        if len(easy_rows) > remaining:
            idx = rng.choice(len(easy_rows), size=remaining, replace=False)
            easy_rows = [easy_rows[int(i)] for i in idx]
    return true_rows + hard_rows + easy_rows


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
