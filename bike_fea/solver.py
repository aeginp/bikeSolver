"""2D Euler-Bernoulli frame solver.

Units: millimetres, newtons, MPa. Three degrees of freedom per node: ux, uy, rotation.
Rotation is counterclockwise positive in the global frame.
Axial force is tension-positive.
A positive end moment sags a member drawn in the +x direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeamElement:
    node_i: int
    node_j: int
    youngs_modulus: float
    area: float
    inertia: float


@dataclass(frozen=True)
class Solution:
    """displacements and reactions are (n_cases, n_nodes, 3).

    End forces are (n_cases, n_elements). Moments are N·mm.
    """

    displacement: np.ndarray
    reactions: np.ndarray
    axial: np.ndarray
    shear: np.ndarray
    moment_i: np.ndarray
    moment_j: np.ndarray


def solve(
    nodes: np.ndarray,
    elements: list[BeamElement],
    fixed: np.ndarray,
    forces: np.ndarray,
) -> Solution:
    """Solve K u = F for one or more load cases.

    `nodes` is (n, 2). `fixed` is (n, 3) bool, True where the DOF is supported.
    `forces` is (n_cases, n, 3): Fx, Fy, moment applied to each node.
    """
    xy = np.asarray(nodes, dtype=float)
    n_nodes = xy.shape[0]
    loads = np.asarray(forces, dtype=float)
    if loads.ndim != 3 or loads.shape[1:] != (n_nodes, 3):
        raise ValueError("forces must have shape (n_cases, n_nodes, 3).")
    support = np.asarray(fixed, dtype=bool)
    if support.shape != (n_nodes, 3):
        raise ValueError("fixed must have shape (n_nodes, 3).")

    n_dof = n_nodes * 3
    stiffness = np.zeros((n_dof, n_dof))
    for element in elements:
        _assemble(stiffness, xy, element)

    free = np.flatnonzero(~support.reshape(-1))
    if free.size == 0:
        raise ValueError("Every degree of freedom is fixed.")
    reduced = stiffness[np.ix_(free, free)]
    applied = loads.reshape(loads.shape[0], n_dof).T
    try:
        free_disp = np.linalg.solve(reduced, applied[free, :])
    except np.linalg.LinAlgError as exc:
        raise ValueError("Stiffness matrix is singular. Check supports and member lengths.") from exc

    displacement = np.zeros((n_dof, loads.shape[0]))
    displacement[free, :] = free_disp
    reactions = stiffness @ displacement - applied

    n_elements = len(elements)
    n_cases = loads.shape[0]
    axial = np.zeros((n_cases, n_elements))
    shear = np.zeros((n_cases, n_elements))
    moment_i = np.zeros((n_cases, n_elements))
    moment_j = np.zeros((n_cases, n_elements))
    for index, element in enumerate(elements):
        ends = _end_forces(xy, element, displacement)
        axial[:, index] = ends[0]
        shear[:, index] = ends[1]
        moment_i[:, index] = ends[2]
        moment_j[:, index] = ends[3]

    return Solution(
        displacement=displacement.T.reshape(n_cases, n_nodes, 3),
        reactions=reactions.T.reshape(n_cases, n_nodes, 3),
        axial=axial,
        shear=shear,
        moment_i=moment_i,
        moment_j=moment_j,
    )


def _local_stiffness(modulus: float, area: float, inertia: float, length: float) -> np.ndarray:
    ea = modulus * area / length
    ei = modulus * inertia
    k = np.zeros((6, 6))
    k[0, 0] = k[3, 3] = ea
    k[0, 3] = k[3, 0] = -ea
    bend = 12.0 * ei / length**3
    couple = 6.0 * ei / length**2
    k[1, 1] = k[4, 4] = bend
    k[1, 4] = k[4, 1] = -bend
    k[1, 2] = k[2, 1] = couple
    k[1, 5] = k[5, 1] = couple
    k[4, 2] = k[2, 4] = -couple
    k[4, 5] = k[5, 4] = -couple
    k[2, 2] = k[5, 5] = 4.0 * ei / length
    k[2, 5] = k[5, 2] = 2.0 * ei / length
    return k


def _transform(cosine: float, sine: float) -> np.ndarray:
    transform = np.zeros((6, 6))
    transform[0, 0] = transform[3, 3] = cosine
    transform[0, 1] = transform[3, 4] = sine
    transform[1, 0] = transform[4, 3] = -sine
    transform[1, 1] = transform[4, 4] = cosine
    transform[2, 2] = transform[5, 5] = 1.0
    return transform


def _element_geometry(xy: np.ndarray, element: BeamElement) -> tuple[float, float, float, np.ndarray]:
    start = xy[element.node_i]
    end = xy[element.node_j]
    delta = end - start
    length = float(np.hypot(delta[0], delta[1]))
    if length < 1e-9:
        raise ValueError(f"Element {element.node_i}-{element.node_j} has zero length.")
    cosine = float(delta[0] / length)
    sine = float(delta[1] / length)
    local = _local_stiffness(element.youngs_modulus, element.area, element.inertia, length)
    transform = _transform(cosine, sine)
    return length, cosine, sine, transform.T @ local @ transform


def _assemble(stiffness: np.ndarray, xy: np.ndarray, element: BeamElement) -> None:
    _, _, _, global_k = _element_geometry(xy, element)
    dofs = _dofs(element)
    stiffness[np.ix_(dofs, dofs)] += global_k


def _end_forces(xy: np.ndarray, element: BeamElement, displacement: np.ndarray) -> np.ndarray:
    """Return (axial, shear, moment_i, moment_j) for each load case. Axial is tension-positive."""
    start = xy[element.node_i]
    end = xy[element.node_j]
    delta = end - start
    length = float(np.hypot(delta[0], delta[1]))
    cosine = float(delta[0] / length)
    sine = float(delta[1] / length)
    local = _local_stiffness(element.youngs_modulus, element.area, element.inertia, length)
    transform = _transform(cosine, sine)
    dofs = _dofs(element)
    local_disp = transform @ displacement[dofs, :]
    local_force = local @ local_disp
    return np.vstack(
        [
            -local_force[0],
            local_force[1],
            local_force[2],
            local_force[5],
        ]
    )


def _dofs(element: BeamElement) -> list[int]:
    i = 3 * element.node_i
    j = 3 * element.node_j
    return [i, i + 1, i + 2, j, j + 1, j + 2]
