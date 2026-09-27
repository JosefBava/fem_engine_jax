"""Hex8 compressible Neo-Hookean block under prescribed shear."""

import numpy as np
import sys
from pathlib import Path

try:
    from ..elements import Hex8
    from ..materials import NeoHookean3D
    from .common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_hex_mesh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fem_engine.elements import Hex8
    from fem_engine.materials import NeoHookean3D
    from fem_engine.examples.common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_hex_mesh


def main():
    mesh = structured_hex_mesh(2, 2, 1, length=2.0, height=2.0, depth=1.0)
    material = NeoHookean3D(youngs_modulus=1.0e4, poisson_ratio=0.45)
    coordinates = np.asarray(mesh.coordinates)
    bottom = np.flatnonzero(np.isclose(coordinates[:, 2], 0.0))
    top = np.flatnonzero(np.isclose(coordinates[:, 2], 1.0))
    constrained = dofs_for_nodes(bottom, (2,)) + dofs_for_nodes([int(bottom[0])], (0, 1))
    prescribed = [0.0] * len(constrained)
    constrained += dofs_for_nodes(top, (0,))
    prescribed += [0.35] * len(top)
    displacement, history = solve_load_steps(
        mesh, Hex8, material, constrained, prescribed, np.zeros(mesh.n_dofs), steps=10
    )
    run_and_plot("hex8_neo_hookean_shear", mesh, displacement, history)


if __name__ == "__main__":
    main()