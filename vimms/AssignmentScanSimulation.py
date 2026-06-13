from __future__ import annotations

from vimms.AssignmentChemicalArtifact import (
    AssignmentChemicalArtifactConfig,
    generate_assignment_chemical_artifact,
    write_assignment_chemical_artifact,
)


AssignmentScanSimulationConfig = AssignmentChemicalArtifactConfig
generate_assignment_scan_artifact = generate_assignment_chemical_artifact
write_assignment_scan_artifact = write_assignment_chemical_artifact


__all__ = [
    "AssignmentChemicalArtifactConfig",
    "generate_assignment_chemical_artifact",
    "write_assignment_chemical_artifact",
    "AssignmentScanSimulationConfig",
    "generate_assignment_scan_artifact",
    "write_assignment_scan_artifact",
]
