"""Prepare a simulated dataset design for untargeted pipeline testing.

This module creates chemicals and an experimental design consisting of two
groups (case and control). The actual generation of mzML files and ground truth
will be added later.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .generate_chemicals import generate_chemicals


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


def main() -> None:
    """Entry point for dataset preparation."""

    chemicals, design = setup_simulation()
    out_dir = Path("./out")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generated {len(chemicals)} chemicals")
    for group, names in design.samples.items():
        print(group, names)
    # TODO: generate mzML files and ground truth tables


if __name__ == "__main__":
    main()
