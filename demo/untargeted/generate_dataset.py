"""Prepare a simulated dataset design for untargeted pipeline testing.

This module creates chemicals and an experimental design consisting of two
groups (case and control). It provides utilities to generate mzML files
for each sample using a simple Top-1 DDA controller and produces a ground
truth table of the underlying peaks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from vimms.Common import POSITIVE, PROTON_MASS
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.rt.column_drift import SimulatedDriftModel

from .generate_chemicals import generate_chemicals
import pandas as pd


@dataclass
class ExperimentalDesign:
    """Simple container describing sample groups."""

    samples: Dict[str, List[str]]


def create_design(n_samples_per_group: int = 5) -> ExperimentalDesign:
    """Return an experimental design with case and control groups."""

    design = {
        "case": [f"case_{i + 1}" for i in range(n_samples_per_group)],
        "control": [f"control_{i + 1}" for i in range(n_samples_per_group)],
    }
    return ExperimentalDesign(samples=design)


def setup_simulation(
    n_chemicals: int = 100, n_samples_per_group: int = 5
) -> Tuple[list, ExperimentalDesign]:
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
    column_params: dict | None = None,
) -> dict:
    """Generate an mzML file for each sample in ``design``.

    Parameters
    ----------
    chemicals:
        List of simulated chemicals.
    design:
        Experimental design describing the sample groups.
    out_dir:
        Directory where the mzML files will be written.
    max_rt:
        Maximum retention time (seconds) for the simulation.
    top_n:
        Number of precursors to fragment in each cycle.
    column_params:
        Optional parameters for :class:`~vimms.rt.column_drift.SimulatedDriftModel`
        specifying ``noise_sd``, ``intercept_params`` and ``linear_params``.

    Returns
    -------
    dict
        Mapping sample names to the chemicals used for simulation.
    """

    sample_chems = {}
    for group, samples in design.samples.items():
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            if column_params is not None:
                drift = SimulatedDriftModel(
                    min_logp=0.0,
                    max_logp=float(max_rt),
                    min_rt=0.0,
                    max_rt=float(max_rt),
                    intercept_mu=column_params.get("intercept_params", (0.0, 5.0))[0],
                    intercept_sd=column_params.get("intercept_params", (0.0, 5.0))[1],
                    slope_mu=1.0 + column_params.get("linear_params", (0.0, 0.001))[0],
                    slope_sd=column_params.get("linear_params", (0.0, 0.001))[1],
                    noise_sd=column_params.get("noise_sd", 0.1),
                )
                col = drift.make_column()
                sample_data = col.get_dataset(chemicals)
            else:
                sample_data = chemicals
            ms = IndependentMassSpectrometer(POSITIVE, sample_data)
            controller = TopNController(
                POSITIVE, top_n, 1, 10, 15, 1000
            )
            env = Environment(
                ms,
                controller,
                0,
                max_rt,
                progress_bar=False,
                out_dir=group_dir,
                out_file=f"{sample}.mzML",
            )
            env.run()
            sample_chems[sample] = sample_data
    return sample_chems


def generate_ground_truth_table(
    chemicals: list,
    design: ExperimentalDesign,
    per_sample_chems: dict | None = None,
) -> pd.DataFrame:
    """Return a table describing the true peaks for each sample.

    Parameters
    ----------
    chemicals:
        Base chemical list used when ``per_sample_chems`` is ``None``.
    design:
        Experimental design describing the samples.
    per_sample_chems:
        Optional mapping of sample name to a list of chemicals with RT drift
        applied. When provided, ground truth retention times are taken from this
        mapping instead of ``chemicals``.
    """

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


def main() -> None:
    """Entry point for dataset preparation."""

    chemicals, design = setup_simulation()
    out_dir = Path("./out")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generated {len(chemicals)} chemicals")
    for group, names in design.samples.items():
        print(group, names)
    generate_mzml_files(chemicals, design, out_dir)
    gt = generate_ground_truth_table(chemicals, design)
    gt.to_csv(out_dir / "ground_truth.csv", index=False)
    write_ground_truth_mgf(chemicals, out_dir / "ground_truth.mgf")


if __name__ == "__main__":
    main()
