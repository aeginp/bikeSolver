"""Frame parameters to node coordinates.

Coordinate system: bottom bracket at the origin, +x forward, +y up.
Lengths are millimetres. Angles are degrees.
The front axle is placed level with the rear axle, at the distance the top tube allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GeometryError(ValueError):
    """Nodes cannot be placed from the given parameters."""


@dataclass(frozen=True)
class FrameGeometry:
    """Independent frame parameters.

    Seat tube angle and head tube angle are measured from horizontal.
    Seat tube is center-to-center, from the bottom bracket to the top tube junction.
    Top tube is the actual tube, from the top tube junction to the head tube top.
    Axle-to-crown runs parallel to the steering axis, from the head tube bottom to the axle.
    Fork offset is perpendicular to the steering axis.
    Saddle height is measured from the bottom bracket along the seat tube axis.
    Stem angle is degrees above horizontal: 0 is straight forward, positive is up.
    """

    wheel_diameter: float
    axle_to_crown: float
    fork_offset: float
    head_tube_angle: float
    head_tube: float
    bb_drop: float
    seat_tube_angle: float
    seat_tube: float
    top_tube: float
    chainstay: float
    saddle_height: float
    seatstay_drop: float = 0.0
    seat_tube_extension: float = 0.0
    stem_length: float = 100.0
    stem_angle: float = -6.0
    spacers: float = 15.0


@dataclass(frozen=True)
class DerivedGeometry:
    stack: float
    reach: float
    effective_top_tube: float
    rear_center: float
    front_center: float
    wheelbase: float
    trail: float
    bb_height: float


@dataclass(frozen=True)
class BuiltGeometry:
    nodes: dict[str, np.ndarray]
    derived: DerivedGeometry


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
    """Place every node. Raise GeometryError when a node cannot be placed."""
    _require_positive("wheel diameter", geo.wheel_diameter)
    _require_positive("axle to crown", geo.axle_to_crown)
    _require_positive("head tube", geo.head_tube)
    _require_positive("seat tube", geo.seat_tube)
    _require_positive("top tube", geo.top_tube)
    _require_positive("chainstay", geo.chainstay)
    _require_positive("saddle height", geo.saddle_height)
    _require_positive("stem length", geo.stem_length)
    if geo.bb_drop <= 0:
        raise GeometryError("BB drop must be positive so the axles sit above the bottom bracket.")
    for name, value in (
        ("Fork offset", geo.fork_offset),
        ("Seatstay drop", geo.seatstay_drop),
        ("Seat tube extension", geo.seat_tube_extension),
        ("Spacers", geo.spacers),
    ):
        if value < 0:
            raise GeometryError(f"{name} cannot be negative.")

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
    if geo.seatstay_drop >= geo.seat_tube:
        raise GeometryError("Seatstay drop puts the seatstay junction at or below the bottom bracket.")
    if geo.saddle_height <= geo.seat_tube + geo.seat_tube_extension:
        raise GeometryError("Saddle height must be above the top of the seat tube.")

    # Seat tube axis, from the bottom bracket toward the saddle.
    seat_axis = np.array([-np.cos(sta), np.sin(sta)])
    bb = np.zeros(2)
    top_tube_junction = geo.seat_tube * seat_axis
    seatstay_junction = top_tube_junction - geo.seatstay_drop * seat_axis
    seat_tube_top = (geo.seat_tube + geo.seat_tube_extension) * seat_axis
    saddle = geo.saddle_height * seat_axis

    rear_axle = np.array([-np.sqrt(geo.chainstay**2 - geo.bb_drop**2), geo.bb_drop])

    # Up the steering axis, and the fork offset direction (forward, perpendicular to it).
    up_axis = np.array([-np.cos(hta), np.sin(hta)])
    offset_dir = np.array([np.sin(hta), np.cos(hta)])
    axle_to_top = (geo.axle_to_crown + geo.head_tube) * up_axis - geo.fork_offset * offset_dir

    head_top_y = geo.bb_drop + axle_to_top[1]
    rise = head_top_y - top_tube_junction[1]
    if geo.top_tube <= abs(rise):
        raise GeometryError(
            f"Top tube ({geo.top_tube:.1f} mm) cannot reach the head tube top, "
            f"which sits {rise:+.1f} mm from the top tube junction."
        )
    head_tube_top = np.array(
        [top_tube_junction[0] + np.sqrt(geo.top_tube**2 - rise**2), head_top_y]
    )
    front_axle = head_tube_top - axle_to_top
    head_tube_bottom = head_tube_top - geo.head_tube * up_axis
    steerer_top = head_tube_top + geo.spacers * up_axis
    stem_rad = np.deg2rad(geo.stem_angle)
    stem_clamp = steerer_top + geo.stem_length * np.array([np.cos(stem_rad), np.sin(stem_rad)])

    if front_axle[0] <= 0:
        raise GeometryError("Front axle lands behind the bottom bracket. Lengthen the top tube.")

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

    # Effective top tube: horizontal distance from the head tube top back to the seat tube axis.
    seat_x_at_stack = head_tube_top[1] / seat_axis[1] * seat_axis[0]
    wheel_radius = geo.wheel_diameter / 2.0
    derived = DerivedGeometry(
        stack=float(head_tube_top[1]),
        reach=float(head_tube_top[0]),
        effective_top_tube=float(head_tube_top[0] - seat_x_at_stack),
        rear_center=float(-rear_axle[0]),
        front_center=float(front_axle[0]),
        wheelbase=float(front_axle[0] - rear_axle[0]),
        trail=float((wheel_radius * np.cos(hta) - geo.fork_offset) / np.sin(hta)),
        bb_height=float(wheel_radius - geo.bb_drop),
    )
    return BuiltGeometry(nodes=nodes, derived=derived)


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise GeometryError(f"{name.capitalize()} must be positive.")
