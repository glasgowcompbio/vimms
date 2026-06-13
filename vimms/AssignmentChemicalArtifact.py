"""Materialize assignment-noise peak tables as VIMMS chemicals and mzML sidecars."""

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
from vimms.Common import POSITIVE, PROTON_MASS
from vimms.Controller.topN import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer


@dataclass(frozen=True)
class AssignmentChemicalArtifactConfig:
    """Configuration for a chemical-backed assignment artifact.

    The assignment scenario is first generated as truth metadata, then each
    truth/artifact feature is materialized as a VIMMS chemical. The model-facing
    peak table remains the original truth-aligned picked table. Optional mzML
    writing is an audit/export side effect and must not mutate model features.
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
    """Build a simple empirical chromatogram centered on one picked feature."""

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
    """Create fragment children for chemicals with available/plausible MS2 support."""

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
    config: AssignmentChemicalArtifactConfig,
    rng: np.random.Generator,
) -> tuple[list[UnknownChemical], pd.DataFrame]:
    """Convert every truth/artifact peak row into a VIMMS UnknownChemical."""

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
        # UnknownChemical stores a neutral mass, but the picked-table contract is
        # already m/z based. Subtracting a proton gives a chemical that emits at
        # the target m/z without changing model-facing values.
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
                "chemical_id": f"assignment_chem_{peak_id}",
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


def _output_path(config: AssignmentChemicalArtifactConfig) -> Path | None:
    """Return an output directory only when mzML writing is requested."""

    if not config.write_mzml:
        return None
    if config.output_dir is None:
        return Path(tempfile.mkdtemp(prefix="vimms_assignment_chemical_"))
    path = Path(config.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_scan_simulation(
    chemicals: list[UnknownChemical],
    config: AssignmentChemicalArtifactConfig,
) -> tuple[Environment, Path | None]:
    """Run VIMMS TopN scan simulation for optional audit/export mzML output."""

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
    """Flatten VIMMS controller scans into a lightweight audit table."""

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


def _empty_scan_summary() -> pd.DataFrame:
    """Return the scan-summary schema used when mzML writing is disabled."""

    return pd.DataFrame(
        columns=["scan_id", "ms_level", "rt_seconds", "n_peaks", "tic"]
    )


def _truth_table_from_peaks(peak_table: pd.DataFrame) -> pd.DataFrame:
    """Extract peak-level truth columns from the model-facing peak table."""

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
    ]
    return peak_table[[c for c in columns if c in peak_table.columns]].copy()


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclass metadata values to JSON-safe objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def generate_assignment_chemical_artifact(
    config: AssignmentChemicalArtifactConfig | None = None,
) -> dict[str, Any]:
    """Generate a chemical-backed assignment artifact bundle.

    The returned peak table is exactly the direct assignment peak table. VIMMS
    chemicals and optional mzML/scan summaries are sidecars for audit and
    downstream export, not a peak re-measurement step.
    """

    if config is None:
        config = AssignmentChemicalArtifactConfig()
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
    if config.write_mzml:
        # Scans are sidecars only. The returned peak table stays bit-identical
        # to the direct picked-table generator used by IonClassifier.
        env, mzml_path = _run_scan_simulation(chemicals, config)
        scan_summary = _scan_summary(env.controller.scans)
    else:
        mzml_path = None
        scan_summary = _empty_scan_summary()
    peak_table = base["peak_table"].copy()
    truth_table = _truth_table_from_peaks(peak_table)
    ion_truth_table = _make_ion_truth_table(
        peak_table,
        base["candidate_table"],
        base["ion_table"],
    )
    metadata = dict(base.get("scenario_metadata", {}))
    source_counts = peak_table["source_type"].value_counts().sort_index().to_dict()
    missing_count = int((ion_truth_table["missing_reason"] == "missing_true_companion").sum())
    if missing_count:
        source_counts["missing_true_companion"] = missing_count
    metadata.update(
        {
            "artifact_mode": "chemical",
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
                "feature_extraction": "not_used_for_model_features",
            },
            "config": _jsonable(asdict(config)),
            "notes": [
                "Assignment artifacts are materialized as VIMMS chemicals.",
                "The model-facing peak table is the original truth-aligned assignment table.",
                "mzML and scan summaries are optional audit sidecars and do not mutate model features.",
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


def write_assignment_chemical_artifact(
    artifact: Mapping[str, Any],
    output_dir: str | Path,
    prefix: str = "assignment",
) -> dict[str, Path]:
    """Write a chemical-backed assignment artifact bundle to disk."""

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
    "AssignmentChemicalArtifactConfig",
    "generate_assignment_chemical_artifact",
    "write_assignment_chemical_artifact",
]
