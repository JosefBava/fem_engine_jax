from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)


class BaseElement(ABC):
    """3D number of dof 3"""

    dofs_per_node = 3

    def __init__(self, coordinates, material):
        # float64
        self.coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
        self.material = material

    @property
    @abstractmethod
    def n_nodes(self):
        raise NotImplementedError

    @abstractmethod
    def get_quadrature_data(self):
        """
       gauss integration points and weights for the element
        """
        raise NotImplementedError

    def compute_energy(self, u_flat, dN_dX_all, weights_all):
        """strain energy of the element given the nodal displacements"""
        u_nodes = u_flat.reshape((self.n_nodes, 3))

        def point_energy(dN_dX, w_detJ):
            #  H_ij = d(u_i) / d(X_j)
            grad_u = u_nodes.T @ dN_dX  # 3x3
            F = jnp.eye(3) + grad_u     # F = I + grad(u)
            
            # Energy density = strain energy per unit volume
            psi = self.material.strain_energy(F)
            return psi * w_detJ

        # Compute the energy at all quadrature points and sum them up
        energies = jax.vmap(point_energy)(dN_dX_all, weights_all)
        return jnp.sum(energies)

    def get_tangent_and_residual(self, u_e):
        """compute the tangent stiffness matrix and residual force vector for the element"""
        u_flat = jnp.asarray(u_e, dtype=jnp.float64).reshape(-1)
        dN_dX_all, weights_all = self.get_quadrature_data()

        # enegry_fn is a function that computes the strain energy of the element given the nodal displacements
        energy_fn = lambda u: self.compute_energy(u, dN_dX_all, weights_all)

        # force balance, the residual is the difference between the external and internal forces. In this case, we assume that the external force is zero, so the residual is simply the negative of the internal force.
        # R = F_ext - F_int
        f_int = jax.grad(energy_fn)(u_flat)
        residual = -f_int

        # (Ke)
        tangent = jax.hessian(energy_fn)(u_flat)

        return tangent, residual