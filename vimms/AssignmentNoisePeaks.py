"""Peak-row generation helpers for assignment-noise scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from vimms.AssignmentNoiseLabels import label_for

if TYPE_CHECKING:
    from vimms.AssignmentNoise import AssignmentNoiseProfile


def _skew(values: np.ndarray) -> float:
    """Return sample skewness with a stable value for near-constant profiles."""

    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    if sd < 1e-8:
        return 0.0
    return float(np.mean(((arr - arr.mean()) / sd) ** 3))


def _beta_score(rng: np.random.Generator, true: bool, fp_prob: float) -> float:
    """Sample an MS2-like support score for true and false candidate evidence."""

    if true:
        return float(rng.beta(9.0, 2.2))
    if rng.random() < fp_prob:
        return float(rng.beta(5.0, 2.4))
    return float(rng.beta(1.2, 8.0))


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
    """Create one normalized picked-peak row with feature and truth metadata."""

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
        # Keep weak residual scores so the feature exists, but make it clear
        # that no useful spectrum should support this row.
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
    """Compute local context features and assign stable peak identifiers."""

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
    """Append unstructured clutter and blank-contaminant background peaks."""

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
    """Append true candidate-role peaks plus split/merged artifacts."""

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
                # Split/shoulder artifacts resemble the parent ion in m/z/RT and
                # abundance profile but remain background labels.
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
                # Merged peaks intentionally have good-looking evidence but are
                # labelled background because the picked feature is not the clean
                # theoretical ion.
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
    """Append wrong-source peaks near theoretical ions and candidate RTs."""

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
    """Append high-quality background peaks that coelute with theoretical ions."""

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
    """Append plausible isotope/adduct clusters for absent or wrong candidates."""

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
            # Reusing a true latent profile makes a decoy cluster look like a
            # coherent biological signal instead of independent random clutter.
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
                # Some decoy clusters get candidate-specific false MS2 support,
                # matching the failure mode the classifier must reject.
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
    """Limit peak count while preserving true ions and hard negative artifacts."""

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
    # True ions are always retained. Hard negatives are retained before easy
    # diffuse clutter so stress profiles remain adversarial after trimming.
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



__all__ = [
    "_add_candidate_ions",
    "_add_coeluting_isobars",
    "_add_diffuse_background",
    "_add_matched_decoy_clusters",
    "_add_structured_interferents",
    "_finalize_peak_table",
    "_trim_peaks",
]
