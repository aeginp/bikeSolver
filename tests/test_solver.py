import numpy as np
import pytest

from bike_fea.solver import BeamElement, solve


def test_cantilever_tip_load():
    length = 1000.0
    modulus = 210_000.0
    area = 100.0
    inertia = 10_000.0
    load = 1000.0
    nodes = np.array([[0.0, 0.0], [length, 0.0]])
    elements = [BeamElement(0, 1, modulus, area, inertia)]
    fixed = np.array([[True, True, True], [False, False, False]])
    forces = np.zeros((1, 2, 3))
    forces[0, 1, 1] = -load

    result = solve(nodes, elements, fixed, forces)

    tip = load * length**3 / (3.0 * modulus * inertia)
    assert result.displacement[0, 1, 1] == pytest.approx(-tip, rel=1e-9)
    assert result.moment_i[0, 0] == pytest.approx(load * length, rel=1e-9)
    assert result.axial[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_simply_supported_midspan_moment():
    length = 2000.0
    modulus = 210_000.0
    inertia = 20_000.0
    load = 800.0
    nodes = np.array([[0.0, 0.0], [length / 2.0, 0.0], [length, 0.0]])
    elements = [
        BeamElement(0, 1, modulus, 100.0, inertia),
        BeamElement(1, 2, modulus, 100.0, inertia),
    ]
    fixed = np.zeros((3, 3), dtype=bool)
    fixed[0, 0] = True
    fixed[0, 1] = True
    fixed[2, 1] = True
    forces = np.zeros((1, 3, 3))
    forces[0, 1, 1] = -load

    result = solve(nodes, elements, fixed, forces)

    assert result.moment_j[0, 0] == pytest.approx(load * length / 4.0, rel=1e-8)
    assert result.reactions[0, 0, 1] == pytest.approx(load / 2.0, rel=1e-8)
    assert result.reactions[0, 2, 1] == pytest.approx(load / 2.0, rel=1e-8)
    assert result.reactions[0, :, 0].sum() == pytest.approx(0.0, abs=1e-6)


def test_axial_tension_is_positive():
    length = 500.0
    modulus = 200_000.0
    area = 50.0
    load = 2500.0
    nodes = np.array([[0.0, 0.0], [length, 0.0]])
    elements = [BeamElement(0, 1, modulus, area, 1.0)]
    fixed = np.array([[True, True, True], [False, True, False]])
    forces = np.zeros((1, 2, 3))
    forces[0, 1, 0] = load

    result = solve(nodes, elements, fixed, forces)

    assert result.displacement[0, 1, 0] == pytest.approx(load * length / (modulus * area))
    assert result.axial[0, 0] == pytest.approx(load)


def test_rigid_frame_reactions_match_statics_and_have_no_horizontal():
    """Pin at the rear base, roller at the front base, two downward loads on the top."""
    width = 1000.0
    height = 500.0
    modulus = 210_000.0
    area = 1.0e6
    inertia = 1.0e12
    rear_load = 1000.0
    front_load = 500.0
    nodes = np.array(
        [
            [0.0, 0.0],
            [0.0, height],
            [width, height],
            [width, 0.0],
        ]
    )
    elements = [
        BeamElement(0, 1, modulus, area, inertia),
        BeamElement(1, 2, modulus, area, inertia),
        BeamElement(2, 3, modulus, area, inertia),
    ]
    fixed = np.zeros((4, 3), dtype=bool)
    fixed[0, 0] = True
    fixed[0, 1] = True
    fixed[3, 1] = True
    forces = np.zeros((2, 4, 3))
    forces[0, 1, 1] = -rear_load
    forces[0, 2, 1] = -front_load
    forces[1, 1, 1] = -2.0 * rear_load
    forces[1, 2, 1] = -2.0 * front_load

    result = solve(nodes, elements, fixed, forces)

    # Moment about the pin: front reaction * width = front_load * width.
    assert result.reactions[0, 3, 1] == pytest.approx(front_load, rel=1e-6)
    assert result.reactions[0, 0, 1] == pytest.approx(rear_load, rel=1e-6)
    assert result.reactions[0, 0, 0] == pytest.approx(0.0, abs=1e-4)
    assert result.reactions[1, 3, 1] == pytest.approx(2.0 * front_load, rel=1e-6)
    assert result.reactions[1, 0, 0] == pytest.approx(0.0, abs=1e-4)
