"""Safety factor and the wall a member needs at its current outside diameter."""

from __future__ import annotations

import math

from bike_fea.materials import Material
from bike_fea.sections import TubeSection

MIN_WALL_MM = 0.05
WALL_TOLERANCE_MM = 0.01


def safety_factor(stress: float, material: Material) -> float:
    """Yield over tensile stress, or compression allowable over compressive stress.

    A positive stress is tension. Compression allowable is used only for a negative stress.
    """
    if stress >= 0:
        limit = material.yield_strength
        demand = stress
    else:
        limit = material.compression_limit()
        demand = abs(stress)
    if limit <= 0:
        raise ValueError(f"{material.name}: strength must be positive.")
    if demand <= 1e-9:
        return math.inf
    return limit / demand


def governing_stress(
    axial: float,
    moment_i: float,
    moment_j: float,
    section: TubeSection,
    material: Material,
) -> tuple[float, float, str]:
    """Peak fiber stress, its safety factor, and which end governs.

    Shear is ignored. Moment is linear, so the peak is at an end.
    """
    best: tuple[float, float, str] | None = None
    axial_stress = axial / section.area
    for location, moment in (("start", moment_i), ("end", moment_j)):
        bending = abs(moment) * section.outer_fiber / section.inertia
        for stress in (axial_stress + bending, axial_stress - bending):
            factor = safety_factor(stress, material)
            candidate = (factor, stress, location)
            if best is None or candidate[0] < best[0] or (
                candidate[0] == best[0] and abs(candidate[1]) > abs(best[1])
            ):
                best = candidate
    assert best is not None
    return best[1], best[0], best[2]


def required_walls(project, active_keys: list[str]) -> dict[str, float]:
    """Thinnest wall at the current OD that meets the target safety factor.

    Each search holds every other member fixed and re-solves.
    """
    from bike_fea.frame import evaluate

    found: dict[str, float] = {}
    for key in active_keys:
        size = project.members[key]
        hi = size.od / 2.0 - MIN_WALL_MM
        if hi <= MIN_WALL_MM:
            raise ValueError(f"{key}: outside diameter is too small to size a wall.")
        found[key] = _bisect_wall(project, key, MIN_WALL_MM, hi, evaluate)
    return found


def _bisect_wall(project, key: str, lo: float, hi: float, evaluate) -> float:
    target = project.target_safety_factor
    if _factor_at(project, key, lo, evaluate) >= target:
        return lo
    if _factor_at(project, key, hi, evaluate) < target:
        return hi
    while hi - lo > WALL_TOLERANCE_MM:
        mid = 0.5 * (lo + hi)
        if _factor_at(project, key, mid, evaluate) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _factor_at(project, key: str, wall: float, evaluate) -> float:
    solved = evaluate(project.with_wall(key, wall))
    return min(case.safety_factor[key] for case in solved.cases)
