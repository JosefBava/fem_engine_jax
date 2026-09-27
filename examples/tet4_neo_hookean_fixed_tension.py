"""Tet4 Neo-Hookean unit cube under prescribed uniaxial tension."""

import numpy as np
import sys
from pathlib import Path

try:
    from ..elements import Tet4
    from ..materials import NeoHookean3D
    from .common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_tet_mesh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fem_engine.elements import Tet4
    from fem_engine.materials import NeoHookean3D
    from fem_engine.examples.common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_tet_mesh


def main():
    mesh = structured_tet_mesh(2, 2, 2, length=1.0, height=1.0, depth=1.0)
    material = NeoHookean3D(youngs_modulus=1.0, poisson_ratio=0.30)
    coordinates = np.asarray(mesh.coordinates)
    left = np.flatnonzero(np.isclose(coordinates[:, 0], 0.0))
    right = np.flatnonzero(np.isclose(coordinates[:, 0], 1.0))
    constrained = dofs_for_nodes(left)
    constrained += dofs_for_nodes(right, (0,))
    prescribed = [0.0] * (3 * len(left)) + [0.02] * len(right)
    displacement, history = solve_load_steps(
        mesh,
        Tet4,
        material,
        constrained,
        prescribed,
        np.zeros(mesh.n_dofs),
        steps=1,
    )
    run_and_plot("tet4_neo_hookean_fixed_tension", mesh, displacement, history)


if __name__ == "__main__":
    main()