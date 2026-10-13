import math
from dataclasses import replace

import numpy as np
import pytest

from bike_fea.geometry import FrameGeometry, GeometryError, build_geometry


def vertical_frame() -> FrameGeometry:
    """90 degree seat and head tubes, so every node is exact by hand."""
    return FrameGeometry(
        wheel_diameter=680.0,
        axle_to_crown=330.0,
        fork_offset=40.0,
        head_tube_angle=90.0,
        head_tube=100.0,
        bb_drop=70.0,
        seat_tube_angle=90.0,
        seat_tube=500.0,
        top_tube=400.0,
        chainstay=400.0,
        saddle_height=700.0,
        seatstay_drop=0.0,
        seat_tube_extension=60.0,
        stem_length=100.0,
        stem_angle=0.0,
        spacers=0.0,
    )


def piuma_pace() -> FrameGeometry:
    """User parameters from the Piuma Pace v2 Fusion sketch."""
    return FrameGeometry(
        wheel_diameter=700.0,
        axle_to_crown=382.0,
        fork_offset=47.0,
        head_tube_angle=70.0,
        head_tube=95.0,
        bb_drop=80.0,
        seat_tube_angle=76.0,
        seat_tube=500.0,
        top_tube=530.0,
        chainstay=420.0,
        saddle_height=690.0,
    )


def test_matches_the_fusion_sketch():
    derived = build_geometry(piuma_pace()).derived
    assert derived.rear_center == pytest.approx(412.311, abs=0.01)
    assert derived.front_center == pytest.approx(615.656, abs=0.01)
    assert derived.bb_height == pytest.approx(270.0)


def test_vertical_frame_places_every_node():
    built = build_geometry(vertical_frame())
    rear_x = -math.sqrt(400.0**2 - 70.0**2)
    derived = built.derived

    assert derived.stack == pytest.approx(500.0)
    assert derived.reach == pytest.approx(400.0)
    assert derived.effective_top_tube == pytest.approx(400.0)
    assert derived.front_center == pytest.approx(440.0)
    assert derived.wheelbase == pytest.approx(440.0 - rear_x)
    assert derived.trail == pytest.approx(-40.0)
    assert derived.bb_height == pytest.approx(270.0)

    nodes = built.nodes
    assert nodes["top_tube_junction"] == pytest.approx([0.0, 500.0])
    assert nodes["seat_tube_top"] == pytest.approx([0.0, 560.0])
    assert nodes["saddle"] == pytest.approx([0.0, 700.0])
    assert nodes["rear_axle"] == pytest.approx([rear_x, 70.0])
    assert nodes["head_tube_top"] == pytest.approx([400.0, 500.0])
    assert nodes["head_tube_bottom"] == pytest.approx([400.0, 400.0])
    assert nodes["front_axle"] == pytest.approx([440.0, 70.0])
    assert nodes["stem_clamp"] == pytest.approx([500.0, 500.0])


def test_axles_are_level_and_top_tube_length_holds():
    geo = piuma_pace()
    nodes = build_geometry(geo).nodes
    assert nodes["front_axle"][1] == pytest.approx(nodes["rear_axle"][1])
    top_tube = np.linalg.norm(nodes["head_tube_top"] - nodes["top_tube_junction"])
    assert top_tube == pytest.approx(geo.top_tube)


def test_trail_and_dropped_seatstays():
    geo = replace(piuma_pace(), seatstay_drop=40.0)
    built = build_geometry(geo)
    hta = math.radians(70.0)
    sta = math.radians(76.0)
    axis = np.array([-math.cos(sta), math.sin(sta)])

    assert built.derived.trail == pytest.approx((350.0 * math.cos(hta) - 47.0) / math.sin(hta))
    assert built.nodes["seatstay_junction"] == pytest.approx(460.0 * axis)


def test_chainstay_shorter_than_bb_drop():
    with pytest.raises(GeometryError, match="shorter than BB drop"):
        build_geometry(replace(vertical_frame(), chainstay=50.0))


def test_top_tube_too_short_to_reach_head_tube():
    with pytest.raises(GeometryError, match="cannot reach"):
        build_geometry(replace(vertical_frame(), seat_tube=100.0, top_tube=200.0))


def test_saddle_must_sit_above_seat_tube_top():
    with pytest.raises(GeometryError, match="Saddle height"):
        build_geometry(replace(vertical_frame(), saddle_height=550.0))
