from dataclasses import dataclass
import jax
import jax.numpy as jnp

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)


@dataclass
class Mesh:
    """ A class representing a finite element mesh in 3D space. It contains the coordinates of the nodes and the connectivity information of the elements."""
    coordinates: jnp.ndarray          #  (n_nodes, 3)
    connectivities: jnp.ndarray       #  (n_elements, n_nodes_per_elem)

    def __post_init__(self):
        # Ensure that the coordinates and connectivities are stored as JAX arrays with appropriate data types
        self.coordinates = jnp.asarray(self.coordinates, dtype=jnp.float64)
        self.connectivities = jnp.asarray(self.connectivities, dtype=jnp.int32)

    @property
    def n_nodes(self) -> int:
        """ Returns the number of nodes in the mesh. """
        return self.coordinates.shape[0]

    @property
    def n_elements(self) -> int:
        """ the number of elements the mesh"""
        return self.connectivities.shape[0]

    @property
    def n_dofs(self) -> int:
        """ """
        return 3 * self.n_nodes

    def element_coordinates(self, index: int) -> jnp.ndarray:
        """and returns the coordinates of the nodes of a specific element given its index."""
        return self.coordinates[self.connectivities[index]]

    def all_element_coordinates(self) -> jnp.ndarray:
        """
        Returns the coordinates of the nodes for all elements in the mesh.
        """
        return self.coordinates[self.connectivities]

    def get_element_dofs(self, index: int) -> jnp.ndarray:
        """
        Given an element index, this method returns the global degrees of freedom (DOFs) associated with that element.
        """
        node_indices = self.connectivities[index]
        # For each node in the element, we have 3 DOFs (x, y, z displacements). The global DOFs for the element are computed by stacking the DOFs for each node.
        dofs = jnp.stack(
            [3 * node_indices, 3 * node_indices + 1, 3 * node_indices + 2],
            axis=-1
        ).reshape(-1)
        return dofs