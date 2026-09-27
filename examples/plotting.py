from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class FEMPlotter:
    """Plot undeformed/deformed 3D meshes colored by displacement magnitude."""

    _edges = {
        4: ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
        8: ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)),
    }

    def plot(self, mesh, displacement, title, filename=None, show=False, scale=None):
        coordinates = np.asarray(mesh.coordinates)
        displacement = np.asarray(displacement).reshape((-1, 3))
        magnitude = np.linalg.norm(displacement, axis=1)
        deformed = coordinates + self._deformation_scale(coordinates, displacement, scale) * displacement

        figure = plt.figure(figsize=(10, 7))
        axis = figure.add_subplot(111, projection="3d")
        self._draw_mesh(axis, coordinates, mesh.connectivities, color="#9aa4b2", alpha=0.45, linestyle="--")
        self._draw_mesh(axis, deformed, mesh.connectivities, color="#0f766e", alpha=0.9, linestyle="-")
        scatter = axis.scatter(deformed[:, 0], deformed[:, 1], deformed[:, 2], c=magnitude, cmap="viridis", s=28)
        figure.colorbar(scatter, ax=axis, pad=0.1, label="Displacement magnitude")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        self._set_equal_aspect(axis, np.vstack((coordinates, deformed)))
        figure.tight_layout()

        if filename is not None:
            filename = Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(filename, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figure)
        return figure

    @staticmethod
    def _deformation_scale(coordinates, displacement, scale):
        if scale is not None:
            return scale
        span = np.ptp(coordinates, axis=0).max()
        maximum_displacement = np.linalg.norm(displacement, axis=1).max()
        return 1.0 if maximum_displacement == 0.0 else 0.25 * span / maximum_displacement

    def _draw_mesh(self, axis, coordinates, connectivities, color, alpha, linestyle):
        for connectivity in np.asarray(connectivities):
            edges = self._edges[len(connectivity)]
            for start, end in edges:
                points = coordinates[[connectivity[start], connectivity[end]]]
                axis.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=alpha, linestyle=linestyle)

    @staticmethod
    def _set_equal_aspect(axis, points):
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = 0.5 * (minimum + maximum)
        radius = 0.5 * np.max(maximum - minimum)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)

    def plot_convergence(self, history, filename=None, show=False):
        """Plot residual norm against the global Newton iteration number."""
        residuals = []
        for step in history:
            residuals.extend(step["history"])
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.semilogy(range(1, len(residuals) + 1), residuals, "o-", color="#0f766e")
        axis.set_xlabel("Newton iteration")
        axis.set_ylabel("Residual norm")
        axis.set_title("Newton-Raphson convergence")
        axis.grid(True, which="both", alpha=0.25)
        figure.tight_layout()
        if filename is not None:
            filename = Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(filename, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figure)
        return figure


class ParaViewExporter:
    """Write FEM states as ParaView-readable VTU files and a PVD collection."""

    _cell_types = {4: 10, 8: 12}

    def write_before_after(self, mesh, displacement, output_dir, name):
        """Write only the undeformed and final deformed states."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        before_name = f"{name}_before.vtu"
        after_name = f"{name}_after.vtu"
        self.write_vtu(mesh, np.zeros_like(np.asarray(displacement)), output_dir / before_name)
        self.write_vtu(mesh, displacement, output_dir / after_name)

        pvd_path = output_dir / f"{name}.pvd"
        pvd_path.write_text(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
            "  <Collection>\n"
            f'    <DataSet timestep="0" group="" part="0" file="{before_name}"/>\n'
            f'    <DataSet timestep="1" group="" part="0" file="{after_name}"/>\n'
            "  </Collection>\n"
            "</VTKFile>\n",
            encoding="utf-8",
        )
        return pvd_path

    def write_collection(self, mesh, states, output_dir, name):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets = []
        for step_index, state in enumerate(states, start=1):
            filename = f"{name}_step_{step_index:03d}.vtu"
            self.write_vtu(mesh, state["displacement"], output_dir / filename)
            datasets.append((state["load_factor"], filename))

        pvd_path = output_dir / f"{name}.pvd"
        entries = "\n".join(
            f'    <DataSet timestep="{time}" group="" part="0" file="{filename}"/>'
            for time, filename in datasets
        )
        pvd_path.write_text(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
            "  <Collection>\n"
            f"{entries}\n"
            "  </Collection>\n"
            "</VTKFile>\n",
            encoding="utf-8",
        )
        return pvd_path

    def write_vtu(self, mesh, displacement, filename):
        coordinates = np.asarray(mesh.coordinates, dtype=float)
        displacement = np.asarray(displacement, dtype=float).reshape((-1, 3))
        deformed = coordinates + displacement
        connectivity = np.asarray(mesh.connectivities, dtype=int)
        n_cells, nodes_per_cell = connectivity.shape
        offsets = np.arange(1, n_cells + 1) * nodes_per_cell
        cell_type = self._cell_types[nodes_per_cell]
        magnitude = np.linalg.norm(displacement, axis=1)

        def values(array):
            return " ".join(f"{value:.16e}" for value in array.reshape(-1))

        points = values(deformed)
        cells = values(connectivity)
        cell_offsets = values(offsets)
        cell_types = values(np.full(n_cells, cell_type, dtype=np.uint8))
        vectors = values(displacement)
        scalars = values(magnitude)
        xml = f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{len(coordinates)}" NumberOfCells="{n_cells}">
      <PointData Scalars="DisplacementMagnitude" Vectors="Displacement">
        <DataArray type="Float64" Name="Displacement" NumberOfComponents="3" format="ascii">{vectors}</DataArray>
        <DataArray type="Float64" Name="DisplacementMagnitude" format="ascii">{scalars}</DataArray>
      </PointData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{points}</DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">{cells}</DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">{cell_offsets}</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">{cell_types}</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
'''
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(xml, encoding="utf-8")