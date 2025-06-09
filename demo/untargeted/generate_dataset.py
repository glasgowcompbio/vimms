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
) -> None:
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
    """

    for group, samples in design.samples.items():
        group_dir = out_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            ms = IndependentMassSpectrometer(POSITIVE, chemicals)
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


def generate_ground_truth_table(
    chemicals: list, design: ExperimentalDesign
) -> pd.DataFrame:
    """Return a table describing the true peaks for each sample."""

    records = []
    for compound_id, chem in enumerate(chemicals):
        mz = chem.mass + PROTON_MASS
        rt_min = chem.get_min_rt()
        rt_max = chem.get_max_rt()
        rt_apex = chem.get_apex_rt()
        for group_samples in design.samples.values():
            for sample in group_samples:
                records.append(
                    {
                        "sample": sample,
                        "compound_id": compound_id,
                        "mz_apex": mz,
                        "rt_apex": rt_apex,
                        "intensity": chem.max_intensity,
                        "mz_min": mz - 0.01,
                        "mz_max": mz + 0.01,
                        "rt_min": rt_min,
                        "rt_max": rt_max,
                    }
                )
    return pd.DataFrame.from_records(records)


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


if __name__ == "__main__":
    main()
