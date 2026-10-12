"""Round tube area and bending inertia."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TubeSection:
    """Section properties. Area and inertia already include both legs of a pair."""

    od: float
    wall: float
    pair: bool
    area: float
    inertia: float
    outer_fiber: float


def tube_section(od: float, wall: float, pair: bool = False) -> TubeSection:
    """Round tube. `od` and `wall` are millimetres. A pair uses 2A and 2I."""
    if od <= 0 or wall <= 0 or wall >= od / 2:
        raise ValueError(f"Wall {wall} mm is not inside outside diameter {od} mm.")
    inner = od - 2.0 * wall
    area = math.pi / 4.0 * (od**2 - inner**2)
    inertia = math.pi / 64.0 * (od**4 - inner**4)
    if pair:
        area *= 2.0
        inertia *= 2.0
    return TubeSection(
        od=od,
        wall=wall,
        pair=pair,
        area=area,
        inertia=inertia,
        outer_fiber=od / 2.0,
    )
