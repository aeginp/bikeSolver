import math
from dataclasses import replace

import numpy as np
import pytest

from bike_fea.geometry import FrameGeometry, GeometryError, build_geometry


def vertical_frame() -> FrameGeometry:
    """90 degree seat and head tubes, so the chart numbers are exact."""
    return FrameGeometry(
        stack=500.0,
        reach=400.0,
        bb_drop=70.0,
        chainstay=400.0,
        seat_tube_angle=90.0,
        seat_tube_length=560.0,
        seat_tube_ctc=500.0,
        seatstay_drop=0.0,
        saddle_height=700.0,
        head_tube_angle=90.0,
        head_tube_length=100.0,
        fork_length=330.0,
        fork_rake=40.0,
        wheel_radius=340.0,
        stem_length=100.0,
        stem_angle=0.0,
        spacers=0.0,
    )


def test_vertical_frame_recovers_chart_numbers():
    built = build_geometry(vertical_frame())
    rear_x = -math.sqrt(400.0**2 - 70.0**2)

    assert built.derived.effective_top_tube == pytest.approx(400.0)
    assert built.derived.front_center == pytest.approx(440.0)
    assert built.derived.wheelbase == pytest.approx(440.0 - rear_x)
    assert built.derived.trail == pytest.approx(-40.0)
    assert built.warnings == ()

    assert built.nodes["bb"] == pytest.approx([0.0, 0.0])
    assert built.nodes["top_tube_junction"] == pytest.approx([0.0, 500.0])
    assert built.nodes["seat_tube_top"] == pytest.approx([0.0, 560.0])
    assert built.nodes["saddle"] == pytest.approx([0.0, 700.0])
    assert built.nodes["rear_axle"] == pytest.approx([rear_x, 70.0])
    assert built.nodes["head_tube_top"] == pytest.approx([400.0, 500.0])
    assert built.nodes["head_tube_bottom"] == pytest.approx([400.0, 400.0])
    assert built.nodes["front_axle"] == pytest.approx([440.0, 70.0])
    assert built.nodes["stem_clamp"] == pytest.approx([500.0, 500.0])


def test_sloping_seat_tube_and_trail():
    built = build_geometry(
        replace(
            vertical_frame(),
            stack=565.0,
            reach=390.0,
            seat_tube_angle=73.5,
            seat_tube_ctc=520.0,
            seatstay_drop=40.0,
            head_tube_angle=73.0,
            fork_rake=45.0,
            wheel_radius=340.0,
        )
    )
    sta = math.radians(73.5)
    ett = 390.0 + 565.0 * math.cos(sta) / math.sin(sta)
    trail = (340.0 * math.cos(math.radians(73.0)) - 45.0) / math.sin(math.radians(73.0))
    junction = 520.0 * np.array([-math.cos(sta), math.sin(sta)])

    assert built.derived.effective_top_tube == pytest.approx(ett)
    assert built.derived.trail == pytest.approx(trail)
    assert built.derived.trail == pytest.approx(56.8916, abs=1e-3)
    assert built.nodes["top_tube_junction"] == pytest.approx(junction)
    assert built.nodes["seatstay_junction"] == pytest.approx(junction - 40.0 * junction / 520.0)


def test_chainstay_shorter_than_bb_drop():
    with pytest.raises(GeometryError, match="shorter than BB drop"):
        build_geometry(replace(vertical_frame(), chainstay=50.0))


def test_junction_must_sit_below_seat_tube_top():
    with pytest.raises(GeometryError, match="center-to-center"):
        build_geometry(replace(vertical_frame(), seat_tube_ctc=560.0))


def test_axle_height_mismatch_warns_but_still_places_nodes():
    built = build_geometry(replace(vertical_frame(), fork_length=300.0))
    assert built.nodes["front_axle"][1] == pytest.approx(100.0)
    assert len(built.warnings) == 1
    assert "axle" in built.warnings[0].lower()
