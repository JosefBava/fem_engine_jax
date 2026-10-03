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


def summarize_load_response(mesh, history, axis=2, top_fraction=1.0):
    """Return a load-step summary with the mean displacement of the loaded surface."""
    coordinates = jnp.asarray(mesh.coordinates)
    axis_values = coordinates[:, axis]
    maximum_value = float(jnp.max(axis_values))
    if top_fraction < 1.0:
        threshold = maximum_value * top_fraction
        top_nodes = jnp.nonzero(axis_values >= threshold)[0]
    else:
        top_nodes = jnp.nonzero(jnp.isclose(axis_values, maximum_value))[0]

    response = []
    for step in history:
        displacement = jnp.asarray(step["displacement"]).reshape((-1, 3))
        mean_top_displacement = float(jnp.mean(displacement[top_nodes, axis]))
        response.append({
            "load_factor": float(step["load_factor"]),
            "mean_top_displacement": mean_top_displacement,
            "max_top_displacement": float(jnp.max(jnp.abs(displacement[top_nodes, axis]))),
        })
    return response


def run_and_plot(name, mesh, displacement, history, output_dir="results", load_axis=2):
    if __package__ in (None, ""):
        from plotting import FEMPlotter, ParaViewExporter
    else:
        from .plotting import FEMPlotter, ParaViewExporter

    output_path = Path(output_dir) / f"{name}.png"
    convergence_path = Path(output_dir) / f"{name}_convergence.png"
    load_path = Path(output_dir) / f"{name}_load_displacement.png"
    gif_path = Path(output_dir) / f"{name}_paraview_steps.gif"
    collection_path = Path(output_dir) / f"{name}_steps.pvd"
    before_after_path = Path(output_dir) / f"{name}_before_after.pvd"
    plotter = FEMPlotter()
    plotter.plot(mesh, displacement, title=name.replace("_", " ").title(), filename=output_path)
    plotter.plot_convergence(history, filename=convergence_path)

    response = summarize_load_response(mesh, history, axis=load_axis)
    plotter.plot_load_displacement(
        response,
        filename=load_path,
        title=f"{name.replace('_', ' ').title()} — load vs. displacement",
    )

    states = [{"load_factor": step["load_factor"], "displacement": step["displacement"]} for step in history]
    ParaViewExporter().write_collection(mesh, states, output_dir, name)
    plotter.plot_load_steps_gif(
        mesh,
        states,
        filename=gif_path,
        title=f"{name.replace('_', ' ').title()} — ParaView load steps",
    )

    ParaViewExporter().write_before_after(mesh, displacement, output_dir, name)
    with (Path(output_dir) / f"{name}_iterations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("load_step", "load_factor", "newton_iteration", "residual_norm"))
        for step_index, step in enumerate(history, start=1):
            residual_text = ", ".join(f"{residual:.3e}" for residual in step["history"])
            print(f"load step {step_index} ({step['load_factor']:.3f}): {residual_text}")
            for iteration, residual in enumerate(step["history"], start=1):
                writer.writerow((step_index, step["load_factor"], iteration, residual))
    with (Path(output_dir) / f"{name}_load_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("load_step", "load_factor", "mean_top_displacement", "max_top_displacement"))
        for step_index, sample in enumerate(response, start=1):
            writer.writerow((step_index, sample["load_factor"], sample["mean_top_displacement"], sample["max_top_displacement"]))

    print(f"{name}: converged in {sum(item['iterations'] for item in history)} Newton iterations")
    print(f"plot: {output_path}")
    print(f"convergence: {convergence_path}")
    print(f"load-displacement: {load_path}")
    print(f"paraview-gif: {gif_path}")
    print(f"paraview-steps: {collection_path}")
    print(f"paraview-before-after: {before_after_path}")
    print(f"iterations: {Path(output_dir) / f'{name}_iterations.csv'}")
    print(f"load-curve: {Path(output_dir) / f'{name}_load_curve.csv'}")