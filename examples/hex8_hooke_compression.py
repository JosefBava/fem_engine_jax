"""Hex8 linear-elastic block in compression."""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

try:
    from ..elements import Hex8
    from ..materials import Hooke3D
    from .common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_hex_mesh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fem_engine.elements import Hex8
    from fem_engine.materials import Hooke3D
    from fem_engine.examples.common import dofs_for_nodes, run_and_plot, solve_load_steps, structured_hex_mesh


def main():
    mesh = structured_hex_mesh(2, 2, 1, length=2.0, height=2.0, depth=1.0)
    material = Hooke3D(youngs_modulus=5.0e5, poisson_ratio=0.30)
    coordinates = np.asarray(mesh.coordinates)
    bottom = np.flatnonzero(np.isclose(coordinates[:, 2], 0.0))
    top = np.flatnonzero(np.isclose(coordinates[:, 2], 1.0))
    constrained = dofs_for_nodes(bottom, (2,)) + dofs_for_nodes([int(bottom[0])], (0, 1))
    force = jnp.zeros(mesh.n_dofs).at[jnp.asarray(dofs_for_nodes(top, (2,)))].set(-200.0 / len(top))
    displacement, history = solve_load_steps(mesh, Hex8, material, constrained, np.zeros(len(constrained)), force, steps=4)
    run_and_plot("hex8_hooke_compression", mesh, displacement, history)


if __name__ == "__main__":
    main()