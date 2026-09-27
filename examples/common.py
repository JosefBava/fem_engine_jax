from pathlib import Path
import csv
import sys

import jax.numpy as jnp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fem_engine.mesh import BoundaryConditions, Mesh
    from fem_engine.solvers.assembler import assemble_system
    from fem_engine.solvers.newton_raphson import NewtonRaphson
else:
    from ..mesh import BoundaryConditions, Mesh
    from ..solvers.assembler import assemble_system
    from ..solvers.newton_raphson import NewtonRaphson


def structured_hex_mesh(nx, ny, nz, length, height, depth):
    """Create a regular Hex8 mesh with x varying fastest."""
    coordinates = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                coordinates.append((length * i / nx, height * j / ny, depth * k / nz))

    def node(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    connectivities = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                connectivities.append(
                    [
                        node(i, j, k), node(i + 1, j, k), node(i + 1, j + 1, k), node(i, j + 1, k),
                        node(i, j, k + 1), node(i + 1, j, k + 1), node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1),
                    ]
                )
    return Mesh(jnp.asarray(coordinates), jnp.asarray(connectivities, dtype=jnp.int32))


def structured_tet_mesh(nx, ny, nz, length, height, depth):
    """Create a regular mesh of six positively oriented Tet4 elements per cell."""
    hex_mesh = structured_hex_mesh(nx, ny, nz, length, height, depth)
    tetrahedra = []
    for hex_nodes in hex_mesh.connectivities:
        a, b, c, d, e, f, g, h = [int(value) for value in hex_nodes]
        tetrahedra.extend([[a, b, c, g], [a, c, d, g], [a, d, h, g], [a, h, e, g], [a, e, f, g], [a, f, b, g]])

    coordinates = hex_mesh.coordinates
    oriented = []
    for tetrahedron in tetrahedra:
        connectivity = jnp.asarray(tetrahedron, dtype=jnp.int32)
        points = coordinates[connectivity]
        signed_volume = jnp.linalg.det((points[1:] - points[0]).T)
        if float(signed_volume) < 0.0:
            connectivity = connectivity.at[2].set(connectivity[3]).at[3].set(connectivity[2])
        oriented.append(connectivity)
    return Mesh(coordinates, jnp.stack(oriented))


def dofs_for_nodes(nodes, components=(0, 1, 2)):
    return [3 * int(node) + component for node in nodes for component in components]


def solve_load_steps(mesh, element_type, material, constrained_dofs, prescribed_values, external_force, steps=5):
    """Solve a static problem with continuation of loads and prescribed values."""
    displacement = jnp.zeros(mesh.n_dofs, dtype=jnp.float64)
    constrained_dofs = jnp.asarray(constrained_dofs, dtype=jnp.int32)
    prescribed_values = jnp.asarray(prescribed_values, dtype=jnp.float64)
    external_force = jnp.asarray(external_force, dtype=jnp.float64)
    elements = [element_type(mesh.element_coordinates(index), material) for index in range(mesh.n_elements)]
    history = []

    for step in range(1, steps + 1):
        factor = step / steps
        boundary_conditions = BoundaryConditions(
            constrained_dofs=constrained_dofs,
            prescribed_values=factor * prescribed_values,
            external_force=factor * external_force,
        )

        def residual_and_tangent(current_displacement):
            return assemble_system(mesh, elements, current_displacement, boundary_conditions)

        displacement, info = NewtonRaphson(tolerance=1e-8, max_iterations=30).solve(
            residual_and_tangent, displacement
        )
        history.append({"load_factor": factor, "displacement": displacement, **info})
        if not info["converged"]:
            raise RuntimeError(f"Load step {step} did not converge: {info}")
    return displacement, history


def run_and_plot(name, mesh, displacement, history, output_dir="results"):
    if __package__ in (None, ""):
        from plotting import FEMPlotter, ParaViewExporter
    else:
        from .plotting import FEMPlotter, ParaViewExporter

    output_path = Path(output_dir) / f"{name}.png"
    plotter = FEMPlotter()
    plotter.plot(mesh, displacement, title=name.replace("_", " ").title(), filename=output_path)
    plotter.plot_convergence(history, filename=Path(output_dir) / f"{name}_convergence.png")
    ParaViewExporter().write_before_after(mesh, displacement, output_dir, name)
    with (Path(output_dir) / f"{name}_iterations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("load_step", "load_factor", "newton_iteration", "residual_norm"))
        for step_index, step in enumerate(history, start=1):
            residual_text = ", ".join(f"{residual:.3e}" for residual in step["history"])
            print(f"load step {step_index} ({step['load_factor']:.3f}): {residual_text}")
            for iteration, residual in enumerate(step["history"], start=1):
                writer.writerow((step_index, step["load_factor"], iteration, residual))
    print(f"{name}: converged in {sum(item['iterations'] for item in history)} Newton iterations")
    print(f"plot: {output_path}")
    print(f"paraview: {Path(output_dir) / f'{name}.pvd'}")
    print(f"iterations: {Path(output_dir) / f'{name}_iterations.csv'}")