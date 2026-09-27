"""Tet4 linear-elastic cantilever under a distributed tip load."""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

try:
    from ..elements import Tet4
    from ..materials import Hooke3D
    from .common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_tet_mesh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fem_engine.elements import Tet4
    from fem_engine.materials import Hooke3D
    from fem_engine.examples.common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_tet_mesh


def main():
    mesh = structured_tet_mesh(3, 1, 1, length=3.0, height=1.0, depth=1.0)
    material = Hooke3D(youngs_modulus=2.0e6, poisson_ratio=0.30)
    coordinates = np.asarray(mesh.coordinates)
    left = np.flatnonzero(np.isclose(coordinates[:, 0], 0.0))
    right = np.flatnonzero(np.isclose(coordinates[:, 0], 3.0))
    constrained = dofs_for_nodes(left)
    force = jnp.zeros(mesh.n_dofs).at[jnp.asarray(dofs_for_nodes(right, (2,)))].set(-100.0 / len(right))
    displacement, history = solve_load_steps(mesh, Tet4, material, constrained, np.zeros(len(constrained)), force, steps=4)
    run_and_plot("tet4_hooke_cantilever", mesh, displacement, history)


if __name__ == "__main__":
    main()