from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vimms.AssignmentNoise import (
    ASSIGNMENT_ROLES,
    AssignmentNoiseProfile,
    AssignmentScenarioConfig,
    _make_ion_truth_table,
    generate_assignment_peak_table,
)
from vimms.Chemicals import MSN, UnknownChemical
from vimms.Chromatograms import EmpiricalChromatogram
from vimms.Common import POSITIVE, PROTON_MASS, ScanParameters
from vimms.Controller.topN import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer


@dataclass(frozen=True)
class AssignmentScanSimulationConfig:
    """Configuration for a scan-level assignment-noise simulation.

    The assignment scenario is first generated as truth metadata, then each
    truth/artifact feature is materialized as a VIMMS chemical and passed
    through the normal scan simulator. The exported feature table is still
    truth-aligned, but m/z, RT, intensity, and MS2 availability are refreshed
    from the simulated scans.
    """

    seed: int = 0
    profile: AssignmentNoiseProfile = field(default_factory=AssignmentNoiseProfile)
    roles: tuple[str, ...] = ASSIGNMENT_ROLES
    present_pattern: tuple[int, ...] | None = None
    output_dir: str | Path | None = None
    prefix: str = "assignment"
    write_mzml: bool = True
    topn_n: int = 20
    isolation_width: float = 0.7
    mz_tol: float = 10.0
    rt_tol: float = 15.0
    min_ms1_intensity: float = 0.0
    force_n: bool = False
    min_chrom_width_seconds: float = 4.0
    rt_padding_seconds: float = 12.0


def _chromatogram_for_peak(
    apex_seconds: float, width_seconds: float
) -> tuple[EmpiricalChromatogram, float]:
    half_width = max(float(width_seconds), 1.0)
    start = max(0.0, float(apex_seconds) - half_width)
    apex = max(float(apex_seconds), start + 0.25)
    end = max(apex + 0.25, float(apex_seconds) + half_width)
    rts = np.array([start, (start + apex) / 2.0, apex, (apex + end) / 2.0, end])
    intensities = np.array([0.02, 0.42, 1.0, 0.42, 0.02])
    mz_offsets = np.zeros_like(rts)
    return EmpiricalChromatogram(rts, mz_offsets, intensities), start


def _make_ms2_children(
    parent: UnknownChemical,
    row: pd.Series,
    rng: np.random.Generator,
) -> list[MSN]:
    score = float(row.get("best_ms2_score", 0.0))
    available = float(row.get("ms2_available", 0.0)) > 0.0
    if not available and score < 0.25:
        return []

    precursor_mz = float(row["mz"])
    low = 70.0
    high = max(low + 5.0, min(precursor_mz - 5.0, precursor_mz * 0.85))
    if high <= low:
        return []

    n_children = int(rng.integers(3, 6))
    fragment_mzs = np.sort(rng.uniform(low, high, size=n_children))
    props = rng.dirichlet(np.ones(n_children))
    scale = max(0.15, min(1.0, score))
    children = [
        MSN(float(mz), 2, float(prop * scale), 1.0, None, parent)
        for mz, prop in zip(fragment_mzs, props)
    ]
    return children


def _materialize_assignment_chemicals(
    peak_table: pd.DataFrame,
    *,
    config: AssignmentScanSimulationConfig,
    rng: np.random.Generator,
) -> tuple[list[UnknownChemical], pd.DataFrame]:
    chemicals: list[UnknownChemical] = []
    truth_rows: list[dict[str, Any]] = []

    for _, row in peak_table.iterrows():
        peak_id = str(row["peak_id"])
        target_mz = float(row["mz"])
        target_rt_seconds = float(row["rt"]) * 60.0
        width_seconds = max(
            config.min_chrom_width_seconds,
            float(row.get("peak_width", 0.08)) * 60.0 * 1.5,
        )
        chromatogram, start_rt = _chromatogram_for_peak(target_rt_seconds, width_seconds)
        neutral_for_unknown = max(1.0, target_mz - PROTON_MASS)
        max_intensity = max(
            float(row.get("intensity", np.exp(row.get("log_intensity", 12.0)))), 1.0
        )
        chemical = UnknownChemical(
            neutral_for_unknown,
            start_rt,
            max_intensity,
            chromatogram,
            children=[],
        )
        children = _make_ms2_children(chemical, row, rng)
        chemical.children = children
        chemicals.append(chemical)
        truth_rows.append(
            {
                "chemical_id": f"scanchem_{peak_id}",
                "peak_id": peak_id,
                "source_type": str(row.get("source_type", "")),
                "candidate_index": int(row.get("candidate_index", -1)),
                "candidate_id": str(row.get("candidate_id", "")),
                "role": str(row.get("role", "")),
                "true_label": int(row.get("true_label", 0)),
                "is_true_ion": int(row.get("true_label", 0) > 0),
                "chemical_backed": 1,
                "spike_noise_used": 0,
                "fragmentable": int(len(children) > 0),
                "target_mz": target_mz,
                "target_rt_seconds": target_rt_seconds,
                "chrom_start_seconds": start_rt,
                "chrom_end_seconds": float(chemical.get_max_rt()),
                "max_intensity": max_intensity,
                "n_ms2_children": len(children),
            }
        )

    return chemicals, pd.DataFrame(truth_rows)


def _output_path(config: AssignmentScanSimulationConfig) -> Path | None:
    if not config.write_mzml:
        return None
    if config.output_dir is None:
        return Path(tempfile.mkdtemp(prefix="vimms_assignment_scan_"))
    path = Path(config.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_scan_simulation(
    chemicals: list[UnknownChemical],
    config: AssignmentScanSimulationConfig,
) -> tuple[Environment, Path | None]:
    out_dir = _output_path(config)
    out_file = f"{config.prefix}.mzML" if out_dir is not None else None
    mass_spec = IndependentMassSpectrometer(
        POSITIVE,
        chemicals,
        spike_noise=None,
        scan_duration={1: 0.6, 2: 0.2},
    )
    controller = TopNController(
        POSITIVE,
        int(config.topn_n),
        float(config.isolation_width),
        float(config.mz_tol),
        float(config.rt_tol),
        float(config.min_ms1_intensity),
        force_N=bool(config.force_n),
    )
    min_time = max(0.0, min(ch.get_min_rt() for ch in chemicals) - config.rt_padding_seconds)
    max_time = max(ch.get_max_rt() for ch in chemicals) + config.rt_padding_seconds
    env = Environment(
        mass_spec,
        controller,
        min_time,
        max_time,
        progress_bar=False,
        out_dir=str(out_dir) if out_dir is not None else None,
        out_file=out_file,
    )
    env.run()
    return env, None if out_dir is None or out_file is None else out_dir / out_file


def _scan_summary(scans: Mapping[int, list[Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ms_level, level_scans in sorted(scans.items()):
        for scan in level_scans:
            rows.append(
                {
                    "scan_id": int(scan.scan_id),
                    "ms_level": int(scan.ms_level),
                    "rt_seconds": float(scan.rt),
                    "n_peaks": int(scan.num_peaks),
                    "tic": float(np.sum(scan.intensities)) if len(scan.intensities) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _precursor_mzs(scan: Any) -> list[float]:
    if scan.scan_params is None:
        return []
    precursors = scan.scan_params.get(ScanParameters.PRECURSOR_MZ) or []
    return [float(precursor.precursor_mz) for precursor in precursors]


def _ms2_scan_counts(
    peak_table: pd.DataFrame,
    ms2_scans: list[Any],
    config: AssignmentScanSimulationConfig,
) -> np.ndarray:
    counts = np.zeros(len(peak_table), dtype=int)
    if not ms2_scans:
        return counts
    for idx, row in peak_table.iterrows():
        target_mz = float(row["mz"])
        target_rt_seconds = float(row["rt"]) * 60.0
        rt_window = max(20.0, float(row.get("peak_width", 0.08)) * 60.0 * 4.0)
        mz_window = max(
            float(config.isolation_width) / 2.0 + 0.01, target_mz * config.mz_tol * 1e-6
        )
        for scan in ms2_scans:
            if abs(float(scan.rt) - target_rt_seconds) > rt_window:
                continue
            if any(
                abs(precursor_mz - target_mz) <= mz_window for precursor_mz in _precursor_mzs(scan)
            ):
                counts[idx] += 1
    return counts


def _refresh_density_features(peaks: pd.DataFrame) -> pd.DataFrame:
    out = peaks.copy()
    mz = out["mz"].to_numpy(dtype=float)
    rt = out["rt"].to_numpy(dtype=float)
    density = []
    for i in range(len(out)):
        density.append(
            int(((np.abs(mz - mz[i]) <= 0.05) & (np.abs(rt - rt[i]) <= 0.25)).sum() - 1)
        )
    out["local_density"] = np.asarray(density, dtype=float)
    return out


def _extract_scan_backed_peak_table(
    peak_table: pd.DataFrame,
    scans: Mapping[int, list[Any]],
    *,
    config: AssignmentScanSimulationConfig,
) -> pd.DataFrame:
    out = peak_table.copy()
    ms1_scans = list(scans.get(1, []))
    ms2_scans = list(scans.get(2, []))
    backed: list[int] = []
    match_counts: list[int] = []
    rt_seconds_values: list[float] = []

    for idx, row in out.iterrows():
        target_mz = float(row["mz"])
        target_rt_seconds = float(row["rt"]) * 60.0
        rt_window = max(3.0, float(row.get("peak_width", 0.08)) * 60.0 * 2.5)
        mz_window = max(target_mz * max(config.profile.sigma_ppm, 2.0) * 4.0e-6, 0.002)
        best: tuple[float, float, float] | None = None
        n_matches = 0

        for scan in ms1_scans:
            if abs(float(scan.rt) - target_rt_seconds) > rt_window:
                continue
            if len(scan.mzs) == 0:
                continue
            mask = np.abs(scan.mzs - target_mz) <= mz_window
            n_matches += int(mask.sum())
            if not mask.any():
                continue
            local_mzs = scan.mzs[mask]
            local_intensities = scan.intensities[mask]
            local_idx = int(np.argmax(local_intensities))
            candidate = (
                float(local_intensities[local_idx]),
                float(local_mzs[local_idx]),
                float(scan.rt),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None:
            backed.append(0)
            match_counts.append(n_matches)
            rt_seconds_values.append(target_rt_seconds)
            out.at[idx, "scan_backed"] = 0
            continue

        intensity, observed_mz, observed_rt_seconds = best
        out.at[idx, "mz"] = observed_mz
        out.at[idx, "rt"] = observed_rt_seconds / 60.0
        out.at[idx, "intensity"] = max(intensity, 1.0)
        out.at[idx, "log_intensity"] = float(np.log(max(intensity, 1.0)))
        out.at[idx, "scan_backed"] = 1
        backed.append(1)
        match_counts.append(n_matches)
        rt_seconds_values.append(observed_rt_seconds)

    out["scan_backed"] = backed
    out["scan_ms1_match_count"] = match_counts
    out["rt_seconds"] = rt_seconds_values

    ms2_counts = _ms2_scan_counts(out, ms2_scans, config)
    out["ms2_scan_count"] = ms2_counts
    out["ms2_available"] = (ms2_counts > 0).astype(float)
    ms2_score_cols = [c for c in out.columns if c.startswith("ms2_score_cand_")]
    for idx, count in enumerate(ms2_counts):
        if count <= 0:
            for col in ms2_score_cols:
                out.at[idx, col] = 0.0
    if ms2_score_cols:
        out["best_ms2_score"] = out[ms2_score_cols].max(axis=1).astype(float)
    out["false_ms2_support"] = (
        (out["true_label"].astype(int) == 0) & (out["best_ms2_score"].astype(float) >= 0.45)
    ).astype(int)
    return _refresh_density_features(out)


def _truth_table_from_peaks(peak_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "scan_backed",
        "ms2_scan_count",
    ]
    return peak_table[[c for c in columns if c in peak_table.columns]].copy()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def generate_assignment_scan_artifact(
    config: AssignmentScanSimulationConfig | None = None,
) -> dict[str, Any]:
    """Generate a VIMMS mzML-backed assignment artifact bundle."""

    if config is None:
        config = AssignmentScanSimulationConfig()
    rng = np.random.default_rng(config.seed + 13_579)
    base = generate_assignment_peak_table(
        AssignmentScenarioConfig(
            seed=int(config.seed),
            profile=config.profile,
            roles=tuple(config.roles),
            present_pattern=config.present_pattern,
        )
    )
    chemicals, chemical_truth = _materialize_assignment_chemicals(
        base["peak_table"],
        config=config,
        rng=rng,
    )
    env, mzml_path = _run_scan_simulation(chemicals, config)
    peak_table = _extract_scan_backed_peak_table(
        base["peak_table"],
        env.controller.scans,
        config=config,
    )
    truth_table = _truth_table_from_peaks(peak_table)
    ion_truth_table = _make_ion_truth_table(
        peak_table,
        base["candidate_table"],
        base["ion_table"],
    )
    scan_summary = _scan_summary(env.controller.scans)
    metadata = dict(base.get("scenario_metadata", {}))
    source_counts = peak_table["source_type"].value_counts().sort_index().to_dict()
    missing_count = int((ion_truth_table["missing_reason"] == "missing_true_companion").sum())
    if missing_count:
        source_counts["missing_true_companion"] = missing_count
    metadata.update(
        {
            "source_mode": "scan",
            "rt_unit": "minutes",
            "scan_rt_unit": "seconds",
            "mzml_path": "" if mzml_path is None else str(mzml_path),
            "n_chemicals": int(len(chemicals)),
            "n_ms1_scans": (
                int((scan_summary["ms_level"] == 1).sum()) if not scan_summary.empty else 0
            ),
            "n_ms2_scans": (
                int((scan_summary["ms_level"] == 2).sum()) if not scan_summary.empty else 0
            ),
            "n_scan_backed_peaks": int(peak_table["scan_backed"].sum()),
            "source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "scan_simulation": {
                "controller": "TopNController",
                "topn_n": int(config.topn_n),
                "isolation_width": float(config.isolation_width),
                "mz_tol": float(config.mz_tol),
                "rt_tol": float(config.rt_tol),
                "min_ms1_intensity": float(config.min_ms1_intensity),
                "force_n": bool(config.force_n),
                "spike_noise_used": False,
                "feature_extraction": "internal_truth_scan_summary",
            },
            "config": _jsonable(asdict(config)),
            "notes": [
                "Assignment artifacts were materialized as VIMMS chemicals and simulated at scan level.",
                "Feature rows are extracted from internal truth and simulated MS1/MS2 scan evidence.",
                "External peak pickers are not used in this first scan-mode bridge.",
                "UniformSpikeNoise is not used because spike peaks are not chemical-backed or fragmentable.",
            ],
        }
    )
    return {
        "peak_table": peak_table,
        "candidate_table": base["candidate_table"],
        "ion_table": base["ion_table"],
        "truth_table": truth_table,
        "ion_truth_table": ion_truth_table,
        "chemical_truth_table": chemical_truth,
        "scan_summary": scan_summary,
        "scenario_metadata": metadata,
        "mzml_path": mzml_path,
    }


def write_assignment_scan_artifact(
    artifact: Mapping[str, Any],
    output_dir: str | Path,
    prefix: str = "assignment",
) -> dict[str, Path]:
    """Write a scan-level assignment artifact bundle to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "peak_table": output_path / f"{prefix}_peaks.csv",
        "candidate_table": output_path / f"{prefix}_candidates.csv",
        "ion_table": output_path / f"{prefix}_ions.csv",
        "truth_table": output_path / f"{prefix}_truth.csv",
        "ion_truth_table": output_path / f"{prefix}_ion_truth.csv",
        "chemical_truth_table": output_path / f"{prefix}_chemical_truth.csv",
        "scan_summary": output_path / f"{prefix}_scan_summary.csv",
        "scenario_metadata": output_path / f"{prefix}_metadata.json",
    }
    for key in (
        "peak_table",
        "candidate_table",
        "ion_table",
        "truth_table",
        "ion_truth_table",
        "chemical_truth_table",
        "scan_summary",
    ):
        table = artifact.get(key)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"artifact[{key!r}] must be a pandas DataFrame")
        table.to_csv(paths[key], index=False)
    paths["scenario_metadata"].write_text(
        json.dumps(artifact.get("scenario_metadata", {}), indent=2),
        encoding="utf-8",
    )
    mzml_path = artifact.get("mzml_path")
    if mzml_path:
        paths["mzml_path"] = Path(mzml_path)
    return paths


__all__ = [
    "AssignmentScanSimulationConfig",
    "generate_assignment_scan_artifact",
    "write_assignment_scan_artifact",
]
