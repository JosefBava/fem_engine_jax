import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class NeoHookean3D:
    """3D compressible Neo-Hookean material model. This class computes the strain energy density given the deformation gradient F. It uses Young's modulus and Poisson's ratio to define the material properties."""

    def __init__(self, youngs_modulus: float, poisson_ratio: float):
        self.youngs_modulus = jnp.float64(youngs_modulus)
        self.poisson_ratio = jnp.float64(poisson_ratio)

        # Compute Lamé parameters (λ and μ) from Young's modulus and Poisson's ratio
        self.shear_modulus = self.youngs_modulus / (
            2.0 * (1.0 + self.poisson_ratio)
        )
        self.lame_lambda = (
            self.youngs_modulus
            * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        )

    def strain_energy(self, F: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the strain energy density (energy per unit volume) for a given deformation gradient F using the compressible Neo-Hookean model.
        """
        # Compute the right Cauchy-Green deformation tensor: C = F^T * F
        C = F.T @ F

        # Compute the first invariant of C: I_C = tr(C)
        I_C = jnp.trace(C)

        # Compute the determinant of F: J = det(F)
        J = jnp.linalg.det(F)

        # Compute the logarithm of J: ln(J)
        ln_J = jnp.log(J)

        # Compute the strain energy density using the Neo-Hookean formulation
        psi = (
            0.5 * self.shear_modulus * (I_C - 3.0 - 2.0 * ln_J)
            + 0.5 * self.lame_lambda * (ln_J**2)
        )
        return psi