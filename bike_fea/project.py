"""Load and save a frame project as TOML.

Load-split numbers in the file are percentages.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, fields
from pathlib import Path

import tomli_w

from bike_fea.frame import MEMBER_CATALOG, FrameProject, MemberSize, default_members
from bike_fea.geometry import FrameGeometry
from bike_fea.loads import LoadCase
from bike_fea.materials import BUILTINS, Material

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "piuma_pace.toml"


def example_project() -> FrameProject:
    """Piuma Pace v2 from the Fusion sketch, in 4130, with three load cases."""
    geometry = FrameGeometry(
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
        seatstay_drop=0.0,
        seat_tube_extension=0.0,
        stem_length=100.0,
        stem_angle=-6.0,
        spacers=15.0,
    )
    return FrameProject(
        geometry=geometry,
        material=BUILTINS["4130"],
        members=default_members(),
        load_cases=(
            LoadCase("seated bump", 75.0, 5.0, 0.60, 0.30, 0.10),
            LoadCase("standing bump", 75.0, 5.0, 0.0, 0.30, 0.70),
            LoadCase("static", 75.0, 1.0, 0.60, 0.30, 0.10),
        ),
        target_safety_factor=2.0,
    )


def load_project(path: str | Path) -> FrameProject:
    text = Path(path).read_bytes()
    return project_from_dict(tomllib.loads(text.decode("utf-8")))


def save_project(project: FrameProject, path: str | Path) -> None:
    Path(path).write_text(tomli_w.dumps(project_to_dict(project)), encoding="utf-8")


def project_to_dict(project: FrameProject) -> dict:
    material = project.material
    if material.name in BUILTINS and material == BUILTINS[material.name]:
        material_doc: dict = {"builtin": material.name}
    else:
        material_doc = {
            "name": material.name,
            "E": material.youngs_modulus,
            "yield": material.yield_strength,
        }
        if material.allowable_compression is not None:
            material_doc["allowable_compression"] = material.allowable_compression
    return {
        "geometry": asdict(project.geometry),
        "material": material_doc,
        "members": {
            key: {"od": project.members[key].od, "wall": project.members[key].wall}
            for key, _display, _pair in MEMBER_CATALOG
        },
        "settings": {"target_safety_factor": project.target_safety_factor},
        "load_cases": [
            {
                "name": case.name,
                "mass": case.mass_kg,
                "bump_g": case.bump_g,
                "saddle": case.saddle * 100.0,
                "bars": case.bars * 100.0,
                "pedals": case.pedals * 100.0,
            }
            for case in project.load_cases
        ],
    }


def project_from_dict(data: dict) -> FrameProject:
    try:
        known = {field.name for field in fields(FrameGeometry)}
        unknown = sorted(set(data["geometry"]) - known)
        if unknown:
            raise ValueError(f"Unknown geometry parameters: {', '.join(unknown)}.")
        geometry = FrameGeometry(**{key: float(value) for key, value in data["geometry"].items()})
        material = _material(data["material"])
        members = {
            key: MemberSize(float(data["members"][key]["od"]), float(data["members"][key]["wall"]))
            for key, _display, _pair in MEMBER_CATALOG
        }
        settings = data.get("settings", {})
        cases = tuple(_load_case(entry) for entry in data["load_cases"])
    except KeyError as exc:
        raise ValueError(f"Project file is missing {exc}.") from exc
    return FrameProject(
        geometry=geometry,
        material=material,
        members=members,
        load_cases=cases,
        target_safety_factor=float(settings.get("target_safety_factor", 2.0)),
    )


def _material(data: dict) -> Material:
    if "builtin" in data:
        name = str(data["builtin"])
        if name not in BUILTINS:
            known = ", ".join(BUILTINS)
            raise ValueError(f"Unknown material {name!r}. Built-ins: {known}.")
        return BUILTINS[name]
    try:
        compression = data.get("allowable_compression")
        return Material(
            name=str(data["name"]),
            youngs_modulus=float(data["E"]),
            yield_strength=float(data["yield"]),
            allowable_compression=None if compression is None else float(compression),
        )
    except KeyError as exc:
        raise ValueError(f"Custom material is missing {exc}.") from exc


def _load_case(entry: dict) -> LoadCase:
    return LoadCase(
        name=str(entry["name"]),
        mass_kg=float(entry["mass"]),
        bump_g=float(entry["bump_g"]),
        saddle=float(entry["saddle"]) / 100.0,
        bars=float(entry["bars"]) / 100.0,
        pedals=float(entry["pedals"]) / 100.0,
    )
