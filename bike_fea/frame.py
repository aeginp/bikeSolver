"""Assemble the bike, solve every load case, and attach required walls."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from bike_fea.geometry import NODE_LABELS, BuiltGeometry, FrameGeometry, build_geometry
from bike_fea.loads import LoadCase
from bike_fea.materials import Material
from bike_fea.sections import TubeSection, tube_section
from bike_fea.sizing import governing_stress
from bike_fea.solver import BeamElement, Solution, solve

# Stem and steerer are not sized. This factor keeps them from bending relative to the frame.
RIGID_STIFFNESS = 200.0
MIN_MEMBER_MM = 1.0

MEMBER_CATALOG: tuple[tuple[str, str, bool], ...] = (
    ("chainstay", "Chainstay", True),
    ("seatstay", "Seatstay", True),
    ("seat_tube", "Seat tube", False),
    ("seat_tube_upper", "Seat tube upper", False),
    ("seat_tube_extension", "Seat tube extension", False),
    ("seatpost", "Seatpost", False),
    ("top_tube", "Top tube", False),
    ("down_tube", "Down tube", False),
    ("head_tube", "Head tube", False),
    ("fork", "Fork", True),
)

DEFAULT_TUBES: dict[str, tuple[float, float]] = {
    "chainstay": (22.2, 0.8),
    "seatstay": (16.0, 0.8),
    "seat_tube": (28.6, 0.9),
    "seat_tube_upper": (28.6, 0.9),
    "seat_tube_extension": (28.6, 0.9),
    "seatpost": (27.2, 2.0),
    "top_tube": (28.6, 0.9),
    "down_tube": (31.8, 0.9),
    "head_tube": (36.0, 1.2),
    "fork": (28.0, 1.0),
}

PAIRS = {key for key, _name, pair in MEMBER_CATALOG if pair}


@dataclass(frozen=True)
class MemberSize:
    od: float
    wall: float


@dataclass(frozen=True)
class FrameProject:
    geometry: FrameGeometry
    material: Material
    members: dict[str, MemberSize]
    load_cases: tuple[LoadCase, ...]
    target_safety_factor: float = 2.0

    def with_wall(self, key: str, wall: float) -> FrameProject:
        updated = dict(self.members)
        updated[key] = MemberSize(self.members[key].od, wall)
        return replace(self, members=updated)


def default_members() -> dict[str, MemberSize]:
    return {key: MemberSize(od, wall) for key, (od, wall) in DEFAULT_TUBES.items()}


@dataclass(frozen=True)
class MemberForce:
    key: str
    display: str
    pair: bool
    length: float
    axial: float
    axial_per_leg: float | None
    moment_start: float
    moment_end: float
    stress: float
    peak_at: str
    safety_factor: float

    @property
    def sense(self) -> str:
        if self.axial > 1e-6:
            return "tension"
        if self.axial < -1e-6:
            return "compression"
        return "zero"


@dataclass(frozen=True)
class CaseResult:
    name: str
    members: tuple[MemberForce, ...]
    safety_factor: dict[str, float]
    rear_fx: float
    rear_fy: float
    front_fy: float
    residual_fx: float
    residual_fy: float
    bb_deflection: float
    bar_deflection: float


@dataclass(frozen=True)
class MemberSummary:
    """Worst load case for one member, plus the wall that meets the target."""

    force: MemberForce
    case_name: str
    required_wall: float


@dataclass(frozen=True)
class SolvedFrame:
    geometry: BuiltGeometry
    cases: tuple[CaseResult, ...]
    displacement: np.ndarray
    node_index: dict[str, int]
    node_names: tuple[str, ...]


@dataclass(frozen=True)
class Analysis:
    geometry: BuiltGeometry
    cases: tuple[CaseResult, ...]
    summaries: tuple[MemberSummary, ...]
    displacement: np.ndarray
    node_index: dict[str, int]
    node_names: tuple[str, ...]


def analyze(project: FrameProject) -> Analysis:
    """Solve the frame and size every active member. Raises if the model cannot be built."""
    from bike_fea.sizing import required_walls

    solved = evaluate(project)
    active = [item.key for item in solved.cases[0].members]
    walls = required_walls(project, active)
    summaries = []
    for key in active:
        governing = min(
            solved.cases,
            key=lambda case: case.safety_factor[key],
        )
        force = next(member for member in governing.members if member.key == key)
        summaries.append(MemberSummary(force, governing.name, walls[key]))
    return Analysis(
        geometry=solved.geometry,
        cases=solved.cases,
        summaries=tuple(summaries),
        displacement=solved.displacement,
        node_index=solved.node_index,
        node_names=solved.node_names,
    )


def evaluate(project: FrameProject) -> SolvedFrame:
    _validate(project)
    built = build_geometry(project.geometry)
    links = _links(built, project)
    names = _node_names(links)
    index = {name: position for position, name in enumerate(names)}
    xy = np.vstack([built.nodes[name] for name in names])
    elements = [
        BeamElement(
            index[link.node_i],
            index[link.node_j],
            link.youngs,
            link.section.area,
            link.section.inertia,
        )
        for link in links
    ]
    fixed = np.zeros((len(names), 3), dtype=bool)
    fixed[index["rear_axle"], 0] = True
    fixed[index["rear_axle"], 1] = True
    fixed[index["front_axle"], 1] = True

    forces = np.zeros((len(project.load_cases), len(names), 3))
    for case_index, case in enumerate(project.load_cases):
        for node_name, downward in case.forces().items():
            forces[case_index, index[node_name], 1] = -downward

    solution = solve(xy, elements, fixed, forces)
    cases = tuple(
        _case_result(case, solution, case_index, links, index, project.material)
        for case_index, case in enumerate(project.load_cases)
    )
    return SolvedFrame(built, cases, solution.displacement, index, tuple(names))


@dataclass(frozen=True)
class _Link:
    key: str
    display: str
    node_i: str
    node_j: str
    pair: bool
    reported: bool
    section: TubeSection
    youngs: float
    length: float


def _links(built: BuiltGeometry, project: FrameProject) -> list[_Link]:
    nodes = built.nodes
    modulus = project.material.youngs_modulus
    links: list[_Link] = []

    def add(
        key: str,
        display: str,
        start: str,
        end: str,
        pair: bool,
        reported: bool,
        section: TubeSection,
    ) -> None:
        length = float(np.linalg.norm(nodes[end] - nodes[start]))
        if length < MIN_MEMBER_MM:
            raise ValueError(f"{display} is shorter than {MIN_MEMBER_MM:.0f} mm.")
        links.append(_Link(key, display, start, end, pair, reported, section, modulus, length))

    def sized(key: str) -> TubeSection:
        size = project.members[key]
        return tube_section(size.od, size.wall, key in PAIRS)

    junction_gap = float(np.linalg.norm(nodes["top_tube_junction"] - nodes["seatstay_junction"]))
    if junction_gap >= MIN_MEMBER_MM:
        seatstay_end = "seatstay_junction"
        add("seat_tube", "Seat tube", "bb", "seatstay_junction", False, True, sized("seat_tube"))
        add(
            "seat_tube_upper",
            "Seat tube upper",
            "seatstay_junction",
            "top_tube_junction",
            False,
            True,
            sized("seat_tube_upper"),
        )
    else:
        # K lies on J. One node, or the seatstay would hang off an unconnected point.
        seatstay_end = "top_tube_junction"
        add("seat_tube", "Seat tube", "bb", "top_tube_junction", False, True, sized("seat_tube"))

    add("chainstay", "Chainstay", "rear_axle", "bb", True, True, sized("chainstay"))
    add("seatstay", "Seatstay", "rear_axle", seatstay_end, True, True, sized("seatstay"))
    add(
        "seat_tube_extension",
        "Seat tube extension",
        "top_tube_junction",
        "seat_tube_top",
        False,
        True,
        sized("seat_tube_extension"),
    )
    add("seatpost", "Seatpost", "seat_tube_top", "saddle", False, True, sized("seatpost"))
    add("top_tube", "Top tube", "top_tube_junction", "head_tube_top", False, True, sized("top_tube"))
    add("down_tube", "Down tube", "bb", "head_tube_bottom", False, True, sized("down_tube"))
    add("head_tube", "Head tube", "head_tube_bottom", "head_tube_top", False, True, sized("head_tube"))
    add("fork", "Fork", "head_tube_bottom", "front_axle", True, True, sized("fork"))

    base = tube_section(28.6, 1.6)
    rigid = TubeSection(
        base.od,
        base.wall,
        False,
        base.area * RIGID_STIFFNESS,
        base.inertia * RIGID_STIFFNESS,
        base.outer_fiber,
    )
    spacer = float(np.linalg.norm(nodes["steerer_top"] - nodes["head_tube_top"]))
    stem_start = "head_tube_top"
    if spacer >= MIN_MEMBER_MM:
        add("steerer", "Steerer", "head_tube_top", "steerer_top", False, False, rigid)
        stem_start = "steerer_top"
    add("stem", "Stem", stem_start, "stem_clamp", False, False, rigid)
    return links


def _node_names(links: list[_Link]) -> list[str]:
    used = {link.node_i for link in links} | {link.node_j for link in links}
    preferred = [
        "rear_axle",
        "bb",
        "seatstay_junction",
        "top_tube_junction",
        "seat_tube_top",
        "saddle",
        "head_tube_bottom",
        "head_tube_top",
        "front_axle",
        "steerer_top",
        "stem_clamp",
    ]
    return [name for name in preferred if name in used]


def _case_result(
    case: LoadCase,
    solution: Solution,
    case_index: int,
    links: list[_Link],
    index: dict[str, int],
    material: Material,
) -> CaseResult:
    members: list[MemberForce] = []
    factors: dict[str, float] = {}
    for element_index, link in enumerate(links):
        if not link.reported:
            continue
        axial = float(solution.axial[case_index, element_index])
        moment_i = float(solution.moment_i[case_index, element_index])
        moment_j = float(solution.moment_j[case_index, element_index])
        stress, factor, location = governing_stress(axial, moment_i, moment_j, link.section, material)
        peak_node = link.node_i if location == "start" else link.node_j
        members.append(
            MemberForce(
                key=link.key,
                display=link.display,
                pair=link.pair,
                length=link.length,
                axial=axial,
                axial_per_leg=(axial / 2.0) if link.pair else None,
                moment_start=moment_i,
                moment_end=moment_j,
                stress=stress,
                peak_at=NODE_LABELS[peak_node],
                safety_factor=factor,
            )
        )
        factors[link.key] = factor

    disp = solution.displacement[case_index]
    applied = sum(case.forces().values())
    reactions = solution.reactions[case_index]
    return CaseResult(
        name=case.name,
        members=tuple(members),
        safety_factor=factors,
        rear_fx=float(reactions[index["rear_axle"], 0]),
        rear_fy=float(reactions[index["rear_axle"], 1]),
        front_fy=float(reactions[index["front_axle"], 1]),
        residual_fx=float(reactions[:, 0].sum()),
        residual_fy=float(reactions[:, 1].sum() - applied),
        bb_deflection=float(disp[index["bb"], 1]),
        bar_deflection=float(disp[index["stem_clamp"], 1] - disp[index["front_axle"], 1]),
    )


def _validate(project: FrameProject) -> None:
    if project.target_safety_factor <= 0:
        raise ValueError("Target safety factor must be positive.")
    if project.material.youngs_modulus <= 0:
        raise ValueError("Young's modulus must be positive.")
    if project.material.yield_strength <= 0:
        raise ValueError("Yield strength must be positive.")
    if project.material.allowable_compression is not None and project.material.allowable_compression <= 0:
        raise ValueError("Compression allowable must be positive.")
    if not project.load_cases:
        raise ValueError("Add at least one load case.")
    missing = [key for key, _name, _pair in MEMBER_CATALOG if key not in project.members]
    if missing:
        raise ValueError(f"Missing tube sizes: {', '.join(missing)}.")
    for case in project.load_cases:
        case.validate()
