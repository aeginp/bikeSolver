"""Side-view frame plot. Members can be colored by a result."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

from bike_fea.frame import Analysis, member_layout
from bike_fea.geometry import NODE_LABELS, BuiltGeometry
from bike_fea.loads import LoadCase

COLOR_BY = ("stress", "axial", "sf")


def draw_frame(
    figure: Figure,
    geometry: BuiltGeometry,
    analysis: Analysis | None = None,
    *,
    color_by: str = "stress",
    case_name: str | None = None,
    deformed: bool = True,
    load_cases: tuple[LoadCase, ...] | None = None,
    warning: str | None = None,
) -> None:
    """Draw into `figure`. `case_name` None colors each member by its worst case."""
    figure.clear()
    ax = figure.add_subplot(111)
    nodes = geometry.nodes
    layout = member_layout(geometry)
    shape_case = _shape_case(analysis, case_name)

    if analysis is not None and deformed and shape_case is not None:
        _draw_deformed(ax, analysis, layout, shape_case)
    _draw_gray(ax, nodes, [item for item in layout if not item.reported])

    if analysis is None:
        _draw_gray(ax, nodes, [item for item in layout if item.reported])
        title = "Geometry"
    else:
        values, label, cmap, norm = _colors(analysis, layout, color_by, case_name)
        segments = [[nodes[item.node_i], nodes[item.node_j]] for item in layout if item.reported]
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=3.0, capstyle="round")
        collection.set_array(np.asarray(values, dtype=float))
        ax.add_collection(collection)
        figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.04, label=label)
        title = "Worst case" if case_name is None else case_name

    if analysis is not None and deformed and shape_case is not None:
        scale = _deformed_scale(analysis, shape_case)
        shape_name = analysis.cases[shape_case].name
        title += f"    deformed x{scale:.0f}"
        if case_name is None:
            title += f" ({shape_name})"
    if analysis is not None and load_cases and shape_case is not None and case_name is not None:
        _load_arrows(ax, nodes, load_cases[shape_case].forces())
    _label_nodes(ax, nodes)
    if warning:
        ax.text(
            0.5,
            0.02,
            warning,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="tab:red",
            fontsize=9,
            wrap=True,
        )
    _fit(ax, nodes)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    figure.tight_layout()


def _fit(ax, nodes: dict[str, np.ndarray]) -> None:
    pts = np.vstack(list(nodes.values()))
    margin = 75.0
    ax.set_xlim(float(pts[:, 0].min()) - margin, float(pts[:, 0].max()) + margin)
    ax.set_ylim(float(pts[:, 1].min()) - margin - 80.0, float(pts[:, 1].max()) + margin)


def _shape_case(analysis: Analysis | None, case_name: str | None) -> int | None:
    if analysis is None or not analysis.cases:
        return None
    if case_name is None:
        return int(np.argmax([abs(case.bb_deflection) for case in analysis.cases]))
    for index, case in enumerate(analysis.cases):
        if case.name == case_name:
            return index
    return 0


def _colors(analysis: Analysis, layout, color_by: str, case_name: str | None):
    by_key = _values_by_key(analysis, color_by, case_name)
    reported = [item for item in layout if item.reported]
    raw = np.array([by_key[item.key] for item in reported], dtype=float)
    if color_by == "axial":
        limit = max(float(np.max(np.abs(raw))), 1.0)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        return raw, "axial force (N), tension +", "coolwarm", norm
    if color_by == "sf":
        finite = raw[np.isfinite(raw)]
        top = float(np.max(finite)) if finite.size else 1.0
        bottom = float(np.min(finite)) if finite.size else 0.0
        if top <= bottom:
            top = bottom + 1.0
        plotted = np.where(np.isfinite(raw), raw, top)
        return plotted, "safety factor", "viridis_r", Normalize(vmin=bottom, vmax=top)
    magnitude = np.abs(raw)
    top = max(float(np.max(magnitude)), 1.0)
    return magnitude, "peak |stress| (MPa)", "plasma", Normalize(vmin=0.0, vmax=top)


def _values_by_key(analysis: Analysis, color_by: str, case_name: str | None) -> dict[str, float]:
    if case_name is None:
        members = [item.force for item in analysis.summaries]
    else:
        case = next(item for item in analysis.cases if item.name == case_name)
        members = list(case.members)
    values = {}
    for member in members:
        if color_by == "axial":
            values[member.key] = member.axial
        elif color_by == "sf":
            values[member.key] = member.safety_factor
        else:
            values[member.key] = abs(member.stress)
    return values


def _draw_gray(ax, nodes, items) -> None:
    for item in items:
        start = nodes[item.node_i]
        end = nodes[item.node_j]
        ax.plot([start[0], end[0]], [start[1], end[1]], color="0.45", linewidth=1.5, solid_capstyle="round")


def _draw_deformed(ax, analysis: Analysis, layout, case_index: int) -> None:
    scale = _deformed_scale(analysis, case_index)
    disp = analysis.displacement[case_index]
    index = analysis.node_index

    def point(name: str) -> np.ndarray:
        return analysis.geometry.nodes[name] + scale * disp[index[name], :2]

    for item in layout:
        if item.node_i not in index or item.node_j not in index:
            continue
        start = point(item.node_i)
        end = point(item.node_j)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )


def _deformed_scale(analysis: Analysis, case_index: int) -> float:
    disp = analysis.displacement[case_index, :, :2]
    peak = float(np.max(np.linalg.norm(disp, axis=1)))
    wheelbase = max(analysis.geometry.derived.wheelbase, 1.0)
    if peak < 1e-9:
        return 1.0
    return 0.12 * wheelbase / peak


def _label_nodes(ax, nodes: dict[str, np.ndarray]) -> None:
    shown = {
        "bb",
        "top_tube_junction",
        "seatstay_junction",
        "seat_tube_top",
        "saddle",
        "head_tube_top",
        "head_tube_bottom",
        "rear_axle",
        "front_axle",
        "stem_clamp",
    }
    offsets = {
        "bb": np.array([-22.0, 12.0]),
        "top_tube_junction": np.array([-16.0, 8.0]),
        "seatstay_junction": np.array([16.0, -10.0]),
        "seat_tube_top": np.array([-16.0, 0.0]),
        "saddle": np.array([-14.0, 8.0]),
        "head_tube_top": np.array([-18.0, 10.0]),
        "head_tube_bottom": np.array([16.0, 0.0]),
        "stem_clamp": np.array([14.0, 10.0]),
        "rear_axle": np.array([-16.0, 8.0]),
        "front_axle": np.array([8.0, -16.0]),
    }
    merged = [
        name
        for name in ("seatstay_junction", "seat_tube_top")
        if name in nodes and float(np.linalg.norm(nodes[name] - nodes["top_tube_junction"])) < 8.0
    ]
    for name, xy in nodes.items():
        if name not in shown or name in merged:
            continue
        label = NODE_LABELS[name]
        if name == "top_tube_junction" and merged:
            label = "/".join([label, *(NODE_LABELS[item] for item in merged)])
        ax.annotate(label, xy, xytext=xy + offsets[name], fontsize=8, ha="center", va="center")


def _load_arrows(ax, nodes: dict[str, np.ndarray], downward: dict[str, float]) -> None:
    peak = max(downward.values()) if downward else 0.0
    if peak <= 0:
        return
    for name, force in downward.items():
        if force <= 0 or name not in nodes:
            continue
        start = nodes[name]
        length = 28.0 + 50.0 * force / peak
        end = start + np.array([0.0, -length])
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        )
        ax.text(end[0] + 8.0, end[1], f"{force:.0f} N", fontsize=7, ha="left", va="center")
