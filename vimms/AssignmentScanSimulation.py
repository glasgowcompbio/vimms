from __future__ import annotations

import warnings

from vimms.AssignmentChemicalArtifact import (
    AssignmentChemicalArtifactConfig,
    generate_assignment_chemical_artifact,
    write_assignment_chemical_artifact,
)


AssignmentScanSimulationConfig = AssignmentChemicalArtifactConfig


def generate_assignment_scan_artifact(*args, **kwargs):
    warnings.warn(
        "generate_assignment_scan_artifact is deprecated; use "
        "generate_assignment_chemical_artifact instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_assignment_chemical_artifact(*args, **kwargs)


def write_assignment_scan_artifact(*args, **kwargs):
    warnings.warn(
        "write_assignment_scan_artifact is deprecated; use "
        "write_assignment_chemical_artifact instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return write_assignment_chemical_artifact(*args, **kwargs)


__all__ = [
    "AssignmentChemicalArtifactConfig",
    "generate_assignment_chemical_artifact",
    "write_assignment_chemical_artifact",
    "AssignmentScanSimulationConfig",
    "generate_assignment_scan_artifact",
    "write_assignment_scan_artifact",
]
