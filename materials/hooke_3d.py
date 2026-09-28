import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Hooke3D:
    """3D linear elastic material model based on Hooke's law. This class computes the strain energy density given the deformation gradient F. It uses Young's modulus and Poisson's ratio to define the material properties."""

    def __init__(self, youngs_modulus: float, poisson_ratio: float):
        self.youngs_modulus = jnp.float64(youngs_modulus)
        self.poisson_ratio = jnp.float64(poisson_ratio)

        # Compute Lamé parameters (λ and μ) from Young's modulus and Poisson's ratio
        self.lame_lambda = (
            self.youngs_modulus
            * self.poisson_ratio
            / ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio))
        )
        self.shear_modulus = self.youngs_modulus / (
            2.0 * (1.0 + self.poisson_ratio)
        )

    def strain_energy(self, F):
        """
        Compute the strain energy density (energy per unit volume) for a given deformation gradient F using the linear elastic Hooke's law.
        """
        #  H = F - I
        H = F - jnp.eye(3, dtype=F.dtype)

        #  epsilon = 0.5 * (H + H^T)
        strain = 0.5 * (H + H.T)

        #  0.5* lambda * (tr(eps))^2 + mu * tr(eps^2)
        trace_strain = jnp.trace(strain)
        energy_density = (
            0.5 * self.lame_lambda * (trace_strain**2)
            + self.shear_modulus * jnp.sum(strain * strain)
        )
        return energy_density