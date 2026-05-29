import warnings

import numpy as np

from vimms.Common import NATURAL_ISOTOPES


def get_isotope_distribution(
    formula,
    total_proportion,
    min_prob=1e-12,
    max_peaks=None,
    max_states=4000,
    mass_precision=8,
):
    distribution = [(0.0, 1.0)]
    monoisotope_log_prob = 0.0
    for element, count in formula.atoms.items():
        if count <= 0:
            continue
        isotopes = NATURAL_ISOTOPES.get(element)
        if not isotopes or len(isotopes) == 1:
            continue
        monoisotope_log_prob += count * np.log(isotopes[0][1])
        mono_mass = isotopes[0][0]
        base_distribution = [(mass - mono_mass, abundance) for mass, abundance in isotopes]
        element_distribution = _power_distribution(
            base_distribution,
            count,
            min_prob=min_prob,
            max_states=max_states,
            mass_precision=mass_precision,
        )
        distribution = _convolve_distributions(
            distribution,
            element_distribution,
            min_prob=min_prob,
            max_states=max_states,
            mass_precision=mass_precision,
        )

    distribution = _preserve_monoisotope(
        distribution, monoisotope_log_prob, min_prob, mass_precision
    )
    selected, cumulative = _select_distribution_peaks(
        distribution, total_proportion, max_peaks
    )
    if max_peaks is not None and cumulative < total_proportion:
        warnings.warn(
            "max_peaks prevented isotope generation from reaching total_proportion; "
            "the truncated isotope envelope will be renormalised.",
            RuntimeWarning,
            stacklevel=2,
        )

    total = sum(prob for _, prob in selected)
    if total == 0:
        return [(0.0, 1.0)]
    return [(shift, prob / total) for shift, prob in selected]


def _preserve_monoisotope(distribution, monoisotope_log_prob, min_prob, mass_precision):
    monoisotope_shift = round(0.0, mass_precision)
    monoisotope_prob = float(np.exp(monoisotope_log_prob))
    distribution_dict = dict(distribution)
    distribution_dict[monoisotope_shift] = monoisotope_prob
    distribution = list(distribution_dict.items())
    distribution = [
        (shift, prob)
        for shift, prob in distribution
        if shift == monoisotope_shift or prob >= min_prob
    ]
    distribution.sort(key=lambda x: x[0])
    return distribution


def _select_distribution_peaks(distribution, total_proportion, max_peaks):
    selected = []
    cumulative = 0.0
    for mass_shift, prob in distribution:
        selected.append((mass_shift, prob))
        cumulative += prob
        reached_total = cumulative >= total_proportion
        reached_cap = max_peaks is not None and len(selected) >= max_peaks
        if reached_total or reached_cap:
            break
    return selected, cumulative


def _power_distribution(base_distribution, count, min_prob, max_states, mass_precision):
    if count == 1:
        return base_distribution
    result = [(0.0, 1.0)]
    power = base_distribution
    remaining = count
    while remaining > 0:
        if remaining % 2 == 1:
            result = _convolve_distributions(
                result,
                power,
                min_prob=min_prob,
                max_states=max_states,
                mass_precision=mass_precision,
            )
        remaining //= 2
        if remaining:
            power = _convolve_distributions(
                power,
                power,
                min_prob=min_prob,
                max_states=max_states,
                mass_precision=mass_precision,
            )
    return result


def _convolve_distributions(left, right, min_prob, max_states, mass_precision):
    new_distribution = {}
    for left_shift, left_prob in left:
        for right_shift, right_prob in right:
            prob = left_prob * right_prob
            if prob < min_prob:
                continue
            shift = left_shift + right_shift
            key = round(shift, mass_precision)
            new_distribution[key] = new_distribution.get(key, 0.0) + prob
    if not new_distribution:
        return []
    distribution = list(new_distribution.items())
    if len(distribution) > max_states:
        distribution.sort(key=lambda x: x[1], reverse=True)
        distribution = distribution[:max_states]
    return distribution
