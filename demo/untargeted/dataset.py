from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from vimms.Common import POSITIVE, PROTON_MASS
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.ColumnDrift import SimulatedDriftModel

from .generate_chemicals import generate_chemicals


@dataclass
class ExperimentalDesign:
    """Simple container describing sample groups."""

    samples: Dict[str, List[str]]


@dataclass
class UntargetedDataset:
    """Container describing an untargeted dataset."""

    mzmls: Dict[str, Path]
    design: ExperimentalDesign
    ground_truth: Optional[pd.DataFrame] = None
    library_path: Optional[Path] = None


def create_design(n_samples_per_group: int = 5) -> ExperimentalDesign:
    """Return an experimental design with case and control groups."""

    design = {
        "case": [f"case_{i + 1}" for i in range(n_samples_per_group)],
        "control": [f"control_{i + 1}" for i in range(n_samples_per_group)],
    }
    return ExperimentalDesign(samples=design)


def setup_simulation(n_chemicals: int = 100, n_samples_per_group: int = 5):
    """Generate chemicals and an experimental design."""

    chemicals = generate_chemicals(n_chemicals)
    design = create_design(n_samples_per_group)
    return chemicals, design


def generate_mzml_files(
    chemicals: list,
    design: ExperimentalDesign,
    out_dir: Path,
    max_rt: int = 180,
    top_n: int = 1,
    column_params=None,
) -> tuple[Dict[str, Path], Dict[str, list]]:
    """Generate an mzML file for each sample in ``design``."""

    sample_paths: Dict[str, Path] = {}
    sample_chems: Dict[str, list] = {}
    for group, samples in design.samples.items():
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            if column_params is not None:
                drift = SimulatedDriftModel(
                    intercept_mu=column_params.get("intercept_params", (0.0, 5.0))[0],
                    intercept_sd=column_params.get("intercept_params", (0.0, 5.0))[1],
                    slope_mu=1.0 + column_params.get("linear_params", (0.0, 0.001))[0],
                    slope_sd=column_params.get("linear_params", (0.0, 0.001))[1],
                    noise_sd=column_params.get("noise_sd", 0.1),
                )
                col = drift.make_column(chemicals)
                sample_data = col.get_dataset()
            else:
                sample_data = chemicals
            ms = IndependentMassSpectrometer(POSITIVE, sample_data)
            controller = TopNController(POSITIVE, top_n, 1, 10, 15, 1000)
            out_file = group_dir / f"{sample}.mzML"
            env = Environment(
                ms,
                controller,
                0,
                max_rt,
                progress_bar=False,
                out_dir=group_dir,
                out_file=out_file.name,
            )
            env.run()
            sample_paths[sample] = out_file
            sample_chems[sample] = sample_data
    return sample_paths, sample_chems


def generate_ground_truth_table(
    chemicals: list,
    design: ExperimentalDesign,
    per_sample_chems: Optional[Dict[str, list]] = None,
) -> pd.DataFrame:
    """Return a table describing the true peaks for each sample."""

    records = []
    for compound_id, chem in enumerate(chemicals):
        mz = chem.mass + PROTON_MASS
        base_rt_min = chem.get_min_rt()
        base_rt_max = chem.get_max_rt()
        for group_samples in design.samples.values():
            for sample in group_samples:
                s_chem = chem if per_sample_chems is None else per_sample_chems[sample][compound_id]
                rt_apex = s_chem.get_apex_rt()
                rt_min = s_chem.get_min_rt() if per_sample_chems else base_rt_min
                rt_max = s_chem.get_max_rt() if per_sample_chems else base_rt_max
                records.append(
                    {
                        "sample": sample,
                        "compound_id": compound_id,
                        "mz_apex": mz,
                        "rt_apex": rt_apex,
                        "intensity": s_chem.max_intensity,
                        "mz_min": mz - 0.01,
                        "mz_max": mz + 0.01,
                        "rt_min": rt_min,
                        "rt_max": rt_max,
                    }
                )
    return pd.DataFrame.from_records(records)


def write_ground_truth_mgf(chemicals: list, out_file: Path) -> None:
    """Write a simple MGF file of MS2 spectra for each chemical."""

    with open(out_file, "w") as f:
        for compound_id, chem in enumerate(chemicals):
            if not chem.children:
                continue
            pepmass = chem.mass + PROTON_MASS
            rt_apex = chem.get_apex_rt()

            f.write("BEGIN IONS\n")
            f.write(f"TITLE=compound_{compound_id}\n")
            f.write(f"PEPMASS={pepmass:.5f}\n")
            f.write("CHARGE=1+\n")
            f.write(f"RTINSECONDS={rt_apex:.2f}\n")
            for frag in chem.children:
                mz = frag.isotopes[0][0]
                intensity = chem.max_intensity * frag.prop_ms2_mass
                f.write(f"{mz:.5f} {intensity:.1f}\n")
            f.write("END IONS\n")


def generate_synthetic_dataset(
    out_dir: Path,
    n_chemicals: int = 100,
    n_samples_per_group: int = 5,
    mzml_max_rt: int = 180,
    top_n: int = 1,
    use_rt_noise: bool = False,
    noise_sd: float = 0.1,
    intercept_params: tuple[float, float] = (0.0, 5.0),
    linear_params: tuple[float, float] = (0.0, 0.001),
) -> UntargetedDataset:
    """Create a synthetic dataset and return an :class:`UntargetedDataset`."""

    out_dir.mkdir(parents=True, exist_ok=True)
    chemicals, design = setup_simulation(n_chemicals, n_samples_per_group)

    column_params = None
    if use_rt_noise:
        column_params = {
            "noise_sd": noise_sd,
            "intercept_params": intercept_params,
            "linear_params": linear_params,
        }

    mzml_paths, per_sample_chems = generate_mzml_files(
        chemicals,
        design,
        out_dir,
        max_rt=mzml_max_rt,
        top_n=top_n,
        column_params=column_params,
    )

    gt = generate_ground_truth_table(
        chemicals, design, per_sample_chems if use_rt_noise else None
    )
    gt_file = out_dir / "ground_truth.csv"
    gt.to_csv(gt_file, index=False)

    mgf_file = out_dir / "ground_truth.mgf"
    write_ground_truth_mgf(chemicals, mgf_file)

    return UntargetedDataset(mzml_paths, design, ground_truth=gt, library_path=mgf_file)


def load_dataset(
    mzml_dir: Path,
    ground_truth_file: Optional[Path] = None,
    library_path: Optional[Path] = None,
) -> UntargetedDataset:
    """Load an existing dataset from ``mzml_dir``."""

    design: Dict[str, List[str]] = {}
    mzmls: Dict[str, Path] = {}
    for group_dir in mzml_dir.iterdir():
        if not group_dir.is_dir():
            continue
        samples: List[str] = []
        for f in sorted(group_dir.glob("*.mzML")):
            sample_name = f.stem
            samples.append(sample_name)
            mzmls[sample_name] = f
        if samples:
            design[group_dir.name] = samples

    gt = pd.read_csv(ground_truth_file) if ground_truth_file else None
    return UntargetedDataset(mzmls, ExperimentalDesign(design), gt, library_path)


if __name__ == "__main__":
    generate_synthetic_dataset(Path("./out"))
