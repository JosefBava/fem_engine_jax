import jax
import jax.numpy as jnp
try:
    from .base_element import BaseElement
except ImportError:
    from base_element import BaseElement

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)


class Tet4(BaseElement):
    """ 4-node tetrahedral element for 3D problems. Each node has 3 degrees of freedom (DOF) corresponding to displacements in the x, y, and z directions. The element uses a single-point integration scheme for numerical integration."""

    n_nodes = 4

    def get_quadrature_data(self):
        """
        Returns the derivatives of the shape functions with respect to the physical coordinates and the weights for single-point integration in a tetrahedral element.
            dN_dX_all: Array of shape (1, 4, 3) containing the derivatives of the shape functions with respect to the physical coordinates at the single quadrature point.
            weights_all: Array of shape (1,) containing the weight for single-point integration, which is the volume of the tetrahedral element.
        """
        # (4, 3)
        dN_dxi = jnp.array(
            [
                [-1.0, -1.0, -1.0],
                [ 1.0,  0.0,  0.0],
                [ 0.0,  1.0,  0.0],
                [ 0.0,  0.0,  1.0],
            ],
            dtype=self.coordinates.dtype,
        )

        # J = X^T @ dN_dxi 
        jacobian = self.coordinates.T @ dN_dxi
        det_j = jnp.linalg.det(jacobian)

        #  dN_dX = dN_dxi @ J^-1 (4*3)
        inv_j = jnp.linalg.inv(jacobian)
        dN_dX = dN_dxi @ inv_j

        #  V = det(J) / 6.0
        volume = det_j / 6.0

        # Return the derivatives of the shape functions with respect to physical coordinates and the weights for single-point integration
        dN_dX_all = jnp.expand_dims(dN_dX, axis=0)  # (1, 4, 3)
        weights_all = jnp.array([volume], dtype=self.coordinates.dtype)  # (1,)

        return dN_dX_all, weights_all

    def _energy_density(self, grad_u):
        """ compute the energy density (strain energy per unit volume) given the gradient of the displacement field"""
        F = jnp.eye(3, dtype=grad_u.dtype) + grad_u
        return self.material.strain_energy(F)