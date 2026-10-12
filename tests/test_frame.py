import math

import pytest

from bike_fea.frame import FrameProject, analyze, default_members, evaluate
from bike_fea.loads import LoadCase
from bike_fea.materials import BUILTINS, Material
from bike_fea.sizing import safety_factor
from tests.test_geometry import vertical_frame


def _project(bump_g: float = 5.0, material=None, drop: float = 0.0, cases=None) -> FrameProject:
    geometry = vertical_frame() if drop == 0 else vertical_frame().__class__(
        **{**vertical_frame().__dict__, "seatstay_drop": drop}
    )
    if cases is None:
        cases = (LoadCase("bump", 75.0, bump_g, 0.6, 0.3, 0.1),)
    return FrameProject(
        geometry=geometry,
        material=material or BUILTINS["4130"],
        members=default_members(),
        load_cases=cases,
        target_safety_factor=2.0,
    )


def test_reactions_match_statics_with_no_horizontal_force():
    project = _project()
    result = analyze(project)
    case = result.cases[0]
    total = project.load_cases[0].total_force
    nodes = result.geometry.nodes
    x_rear = nodes["rear_axle"][0]
    wheelbase = nodes["front_axle"][0] - x_rear
    shares = project.load_cases[0].forces()
    xs = {
        "saddle": nodes["saddle"][0],
        "stem_clamp": nodes["stem_clamp"][0],
        "bb": nodes["bb"][0],
    }
    front = sum(force * (xs[name] - x_rear) for name, force in shares.items()) / wheelbase

    assert case.front_fy == pytest.approx(front, rel=1e-6)
    assert case.rear_fy == pytest.approx(total - front, rel=1e-6)
    assert case.rear_fx == pytest.approx(0.0, abs=1e-4)
    assert case.residual_fx == pytest.approx(0.0, abs=1e-4)
    assert case.residual_fy == pytest.approx(0.0, abs=1e-4)
    assert case.bb_deflection < 0
    assert math.isfinite(case.bar_deflection)


def test_pair_axial_is_reported_per_leg():
    chainstay = next(member for member in analyze(_project()).cases[0].members if member.key == "chainstay")
    assert chainstay.pair
    assert chainstay.axial_per_leg == pytest.approx(chainstay.axial / 2.0)


def test_dropped_seatstays_split_the_seat_tube():
    level = {item.force.key for item in analyze(_project()).summaries}
    dropped = {item.force.key for item in analyze(_project(drop=40.0)).summaries}
    assert "seat_tube_upper" not in level
    assert "seat_tube_upper" in dropped


def test_compression_allowable_applies_only_when_peak_stress_is_negative():
    tension = Material("carbon", 205_000.0, 500.0)
    assert safety_factor(100.0, tension) == pytest.approx(5.0)
    limited = Material("carbon", 205_000.0, 500.0, allowable_compression=200.0)
    assert safety_factor(-100.0, limited) == pytest.approx(2.0)
    assert safety_factor(100.0, limited) == pytest.approx(5.0)

    full = analyze(_project(material=Material("carbon", 205_000.0, 460.0, allowable_compression=460.0)))
    low = analyze(_project(material=Material("carbon", 205_000.0, 460.0, allowable_compression=100.0)))
    saw_compression = False
    for left, right in zip(full.summaries, low.summaries, strict=True):
        assert right.force.safety_factor <= left.force.safety_factor + 1e-9
        if left.force.stress >= 0 and right.force.stress >= 0:
            assert right.force.safety_factor == pytest.approx(left.force.safety_factor)
        if right.force.stress < 0 and right.force.safety_factor < left.force.safety_factor:
            saw_compression = True
    assert saw_compression


def test_required_wall_meets_target_and_grows_with_load():
    light = analyze(_project(bump_g=2.0))
    heavy_project = _project(bump_g=8.0)
    heavy = analyze(heavy_project)
    grew = False
    for left, right in zip(light.summaries, heavy.summaries, strict=True):
        assert right.required_wall >= left.required_wall - 1e-9
        grew = grew or right.required_wall > left.required_wall + 0.02
    assert grew

    sized = next(
        item
        for item in heavy.summaries
        if 0.1 < item.required_wall < heavy_project.members[item.force.key].od / 2.0 - 0.2
    )
    solved = evaluate(heavy_project.with_wall(sized.force.key, sized.required_wall))
    factor = min(case.safety_factor[sized.force.key] for case in solved.cases)
    assert factor == pytest.approx(2.0, rel=0.05)
    assert factor >= 2.0


def test_worst_case_is_the_heavier_bump():
    project = _project(
        cases=(
            LoadCase("static", 75.0, 1.0, 0.6, 0.3, 0.1),
            LoadCase("bump", 75.0, 5.0, 0.6, 0.3, 0.1),
        )
    )
    result = analyze(project)
    assert all(item.case_name == "bump" for item in result.summaries)
    assert len(result.cases) == 2


def test_bar_load_bends_the_head_tube():
    head = next(member for member in analyze(_project()).cases[0].members if member.key == "head_tube")
    assert max(abs(head.moment_start), abs(head.moment_end)) > 0
    assert math.isfinite(head.safety_factor)
