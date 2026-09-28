from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass
class BoundaryConditions:
    constrained_dofs: jnp.ndarray = field(default_factory=lambda: jnp.empty(0, dtype=jnp.int32))
    prescribed_values: jnp.ndarray = field(default_factory=lambda: jnp.empty(0, dtype=jnp.float64))
    external_force: jnp.ndarray = None

    def apply_to_linear_system(self, K: jnp.ndarray, P: jnp.ndarray):
        if self.constrained_dofs.size == 0:
            return K, P

        dofs = self.constrained_dofs
        vals = self.prescribed_values

        P_mod = P - K[:, dofs] @ vals
        P_mod = P_mod.at[dofs].set(vals)

        K_mod = K.at[dofs, :].set(0.0)
        K_mod = K_mod.at[:, dofs].set(0.0)
        K_mod = K_mod.at[dofs, dofs].set(1.0)
        return K_mod, P_mod

    def apply_to_nonlinear_step(
        self,
        tangent: jnp.ndarray,
        residual: jnp.ndarray,
        u_current: jnp.ndarray
    ):
        if self.constrained_dofs.size == 0:
            return tangent, residual

        dofs = self.constrained_dofs
        vals = self.prescribed_values

        # set the target displacements for the constrained DOFs
        delta_u_target = vals - u_current[dofs]

        # Modify the residual to account for the prescribed displacements
        residual_mod = residual - tangent[:, dofs] @ delta_u_target
        residual_mod = residual_mod.at[dofs].set(delta_u_target)

        # Modify the tangent stiffness matrix to enforce the boundary conditions
        tangent_mod = tangent.at[dofs, :].set(0.0)
        tangent_mod = tangent_mod.at[:, dofs].set(0.0)
        tangent_mod = tangent_mod.at[dofs, dofs].set(1.0)

        return tangent_mod, residual_mod

    def get_external_force(self, n_dofs: int) -> jnp.ndarray:
        if self.external_force is None:
            return jnp.zeros(n_dofs, dtype=jnp.float64)
        return jnp.asarray(self.external_force, dtype=jnp.float64)