"""Chart geometry to node coordinates.

Coordinate system: bottom bracket at the origin, +x forward, +y up.
Lengths are millimetres. Angles are degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Front and rear axle heights differ by more than this when the chart inputs disagree.
AXLE_HEIGHT_TOLERANCE_MM = 1.0


class GeometryError(ValueError):
    """Nodes cannot be placed from the given chart inputs."""


@dataclass(frozen=True)
class FrameGeometry:
    """Independent chart inputs.

    Seat tube angle and head tube angle are measured from horizontal.
    Stem angle is degrees above horizontal: 0 is straight forward, positive is up.
    Saddle height is the distance from the bottom bracket to the saddle along the seat tube axis.
    """

    stack: float
    reach: float
    bb_drop: float
    chainstay: float
    seat_tube_angle: float
    seat_tube_length: float
    seat_tube_ctc: float
    seatstay_drop: float
    saddle_height: float
    head_tube_angle: float
    head_tube_length: float
    fork_length: float
    fork_rake: float
    wheel_radius: float
    stem_length: float
    stem_angle: float
    spacers: float


@dataclass(frozen=True)
class DerivedGeometry:
    effective_top_tube: float
    wheelbase: float
    front_center: float
    trail: float


@dataclass(frozen=True)
class BuiltGeometry:
    nodes: dict[str, np.ndarray]
    derived: DerivedGeometry
    warnings: tuple[str, ...]


NODE_LABELS: dict[str, str] = {
    "bb": "BB",
    "top_tube_junction": "J",
    "seatstay_junction": "K",
    "seat_tube_top": "T",
    "saddle": "S",
    "head_tube_top": "HT top",
    "head_tube_bottom": "HT bot",
    "rear_axle": "rear axle",
    "front_axle": "front axle",
    "steerer_top": "steerer",
    "stem_clamp": "stem",
}


def build_geometry(geo: FrameGeometry) -> BuiltGeometry:
    """Place every node. Raise GeometryError when a node cannot be placed.

    A front axle that does not sit on the rear axle height is a warning.
    The nodes are still returned.
    """
    _require_positive("stack", geo.stack)
    _require_positive("reach", geo.reach)
    _require_positive("chainstay", geo.chainstay)
    _require_positive("seat tube length", geo.seat_tube_length)
    _require_positive("seat tube center-to-center", geo.seat_tube_ctc)
    _require_positive("saddle height", geo.saddle_height)
    _require_positive("head tube length", geo.head_tube_length)
    _require_positive("fork length", geo.fork_length)
    _require_positive("wheel radius", geo.wheel_radius)
    _require_positive("stem length", geo.stem_length)
    if geo.bb_drop <= 0:
        raise GeometryError("BB drop must be positive so the axles sit above the bottom bracket.")
    if geo.seatstay_drop < 0:
        raise GeometryError("Seatstay drop cannot be negative.")
    if geo.spacers < 0:
        raise GeometryError("Spacer stack cannot be negative.")
    if geo.fork_rake < 0:
        raise GeometryError("Fork rake cannot be negative.")

    sta = np.deg2rad(geo.seat_tube_angle)
    hta = np.deg2rad(geo.head_tube_angle)
    if np.sin(sta) <= 0:
        raise GeometryError("Seat tube angle must be between 0 and 180 degrees.")
    if np.sin(hta) <= 0:
        raise GeometryError("Head tube angle must be between 0 and 180 degrees.")

    if geo.chainstay < geo.bb_drop:
        raise GeometryError(
            f"Chainstay ({geo.chainstay:.1f} mm) is shorter than BB drop ({geo.bb_drop:.1f} mm)."
        )
    if geo.seat_tube_ctc >= geo.seat_tube_length:
        raise GeometryError(
            "Seat tube center-to-center must sit below the top of the seat tube."
        )
    if geo.seatstay_drop >= geo.seat_tube_ctc:
        raise GeometryError("Seatstay drop puts the seatstay junction at or below the bottom bracket.")
    if geo.saddle_height <= geo.seat_tube_length:
        raise GeometryError("Saddle height must be above the top of the seat tube.")

    # Seat tube axis, from the bottom bracket toward the saddle.
    seat_axis = np.array([-np.cos(sta), np.sin(sta)])
    bb = np.zeros(2)
    top_tube_junction = geo.seat_tube_ctc * seat_axis
    seatstay_junction = top_tube_junction - geo.seatstay_drop * seat_axis
    seat_tube_top = geo.seat_tube_length * seat_axis
    saddle = geo.saddle_height * seat_axis

    rear_x = -np.sqrt(geo.chainstay**2 - geo.bb_drop**2)
    rear_axle = np.array([rear_x, geo.bb_drop])

    # Down the steering axis, toward the front axle: forward and down.
    down_axis = np.array([np.cos(hta), -np.sin(hta)])
    rake_dir = np.array([np.sin(hta), np.cos(hta)])
    head_tube_top = np.array([geo.reach, geo.stack])
    head_tube_bottom = head_tube_top + geo.head_tube_length * down_axis
    front_axle = head_tube_bottom + geo.fork_length * down_axis + geo.fork_rake * rake_dir
    steerer_top = head_tube_top - geo.spacers * down_axis
    stem_rad = np.deg2rad(geo.stem_angle)
    stem_dir = np.array([np.cos(stem_rad), np.sin(stem_rad)])
    stem_clamp = steerer_top + geo.stem_length * stem_dir

    nodes = {
        "bb": bb,
        "top_tube_junction": top_tube_junction,
        "seatstay_junction": seatstay_junction,
        "seat_tube_top": seat_tube_top,
        "saddle": saddle,
        "head_tube_top": head_tube_top,
        "head_tube_bottom": head_tube_bottom,
        "rear_axle": rear_axle,
        "front_axle": front_axle,
        "steerer_top": steerer_top,
        "stem_clamp": stem_clamp,
    }

    # Effective top tube: horizontal distance from the head tube top to the seat tube axis.
    t_intersect = geo.stack / seat_axis[1]
    effective_top_tube = float(geo.reach - t_intersect * seat_axis[0])
    derived = DerivedGeometry(
        effective_top_tube=effective_top_tube,
        wheelbase=float(front_axle[0] - rear_axle[0]),
        front_center=float(front_axle[0]),
        trail=float((geo.wheel_radius * np.cos(hta) - geo.fork_rake) / np.sin(hta)),
    )

    warnings: list[str] = []
    axle_gap = float(front_axle[1] - rear_axle[1])
    if abs(axle_gap) > AXLE_HEIGHT_TOLERANCE_MM:
        warnings.append(
            f"Front axle is {axle_gap:+.1f} mm from the rear axle height. "
            "Check head tube length, fork length, and rake."
        )

    return BuiltGeometry(nodes=nodes, derived=derived, warnings=tuple(warnings))


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise GeometryError(f"{name.capitalize()} must be positive.")
