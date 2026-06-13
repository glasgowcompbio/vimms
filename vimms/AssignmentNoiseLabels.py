from __future__ import annotations


def label_for(candidate_index: int, role_index: int, n_roles: int) -> int:
    """Map candidate/role coordinates to a peak-assignment class label."""

    return 1 + int(candidate_index) * int(n_roles) + int(role_index)


__all__ = ["label_for"]
