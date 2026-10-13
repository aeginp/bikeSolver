from pathlib import Path

from bike_fea.materials import Material
from bike_fea.project import (
    EXAMPLE_PATH,
    example_project,
    load_project,
    project_from_dict,
    project_to_dict,
    save_project,
)


def test_example_round_trips(tmp_path: Path):
    project = example_project()
    path = tmp_path / "frame.toml"
    save_project(project, path)
    assert load_project(path) == project


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
    assert load_project(EXAMPLE_PATH) == example_project()
