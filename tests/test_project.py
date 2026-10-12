from pathlib import Path

import pytest

from bike_fea.geometry import build_geometry
from bike_fea.materials import Material
from bike_fea.project import example_project, load_project, project_from_dict, project_to_dict, save_project


def test_example_axles_are_level_and_round_trip(tmp_path: Path):
    project = example_project()
    built = build_geometry(project.geometry)
    assert built.warnings == ()

    path = tmp_path / "frame.toml"
    save_project(project, path)
    loaded = load_project(path)
    assert loaded == project


def test_custom_material_round_trip():
    project = example_project()
    custom = project.__class__(
        geometry=project.geometry,
        material=Material("carbon-quasi-iso", 55_000.0, 500.0, allowable_compression=350.0),
        members=project.members,
        load_cases=project.load_cases,
        target_safety_factor=project.target_safety_factor,
    )
    restored = project_from_dict(project_to_dict(custom))
    assert restored.material == custom.material


def test_shipped_example_matches_the_preset():
    path = Path(__file__).resolve().parent.parent / "examples" / "road56_steel.toml"
    assert load_project(path) == example_project()
