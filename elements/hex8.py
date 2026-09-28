import jax
import jax.numpy as jnp

try:
    from .base_element import BaseElement
except ImportError:
    from base_element import BaseElement

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)


class Hex8(BaseElement):
    """ 8-node hexahedral element for 3D problems. Each node has 3 degrees of freedom (DOF) corresponding to displacements in the x, y, and z directions. The element uses Gaussian quadrature for numerical integration."""

    n_nodes = 8

    # Define the signs for the shape function derivatives in natural coordinates (ξ, η, ζ)
    _signs = jnp.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    def get_quadrature_data(self):
        """
        Returns the derivatives of the shape functions with respect to the physical coordinates and the weights for Gaussian quadrature integration.
            dN_dX_all: Array of shape (n_quad_points, n_nodes, 3) containing the derivatives of the shape functions with respect to the physical coordinates at each quadrature point.
            weights_all: Array of shape (n_quad_points,) containing the weights for Gaussian quadrature integration, which are the product of the Gaussian weights and the determinant of the Jacobian at each quadrature point.
        """
        point = 1.0 / jnp.sqrt(3.0)
        # gauss_points are the natural coordinates (ξ, η, ζ) of the 8 Gauss points for 2x2x2 integration in a hexahedral element
        gauss_points = jnp.array(
            [
                [-point, -point, -point],
                [point, -point, -point],
                [-point, point, -point],
                [point, point, -point],
                [-point, -point, point],
                [point, -point, point],
                [-point, point, point],
                [point, point, point],
            ],
            dtype=self.coordinates.dtype,
        )

        def compute_single_gp(xi):
            # shape function derivatives with respect to natural coordinates (ξ, η, ζ)
            dN_dxi = 0.125 * jnp.stack(
                (
                    self._signs[:, 0] * (1.0 + self._signs[:, 1] * xi[1]) * (1.0 + self._signs[:, 2] * xi[2]),
                    self._signs[:, 1] * (1.0 + self._signs[:, 0] * xi[0]) * (1.0 + self._signs[:, 2] * xi[2]),
                    self._signs[:, 2] * (1.0 + self._signs[:, 0] * xi[0]) * (1.0 + self._signs[:, 1] * xi[1]),
                ),
                axis=1,
            )  # (8, 3)

            # J = X^T @ dN_dxi
            jacobian = self.coordinates.T @ dN_dxi
            det_j = jnp.linalg.det(jacobian)

            #  dN_dX = dN_dxi @ J^-1 
            dN_dX = dN_dxi @ jnp.linalg.inv(jacobian)
            return dN_dX, det_j

        # Compute the derivatives of the shape functions with respect to physical coordinates and the weights for all Gauss points
        dN_dX_all, weights_all = jax.vmap(compute_single_gp)(gauss_points)
        return dN_dX_all, weights_all

    def _energy_density(self, grad_u):
        """ compute the energy density (strain energy per unit volume) given the gradient of the displacement field"""
        F = jnp.eye(3, dtype=grad_u.dtype) + grad_u
        return self.material.strain_energy(F)