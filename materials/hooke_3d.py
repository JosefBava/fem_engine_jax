import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Hooke3D:
    """رفتار همسانگرد خطی ۳ بعدی بر اساس کرنش‌های کوچک"""

    def __init__(self, youngs_modulus: float, poisson_ratio: float):
        self.youngs_modulus = jnp.float64(youngs_modulus)
        self.poisson_ratio = jnp.float64(poisson_ratio)

        # محاسبه ثوابت لامه: لامبدا (Lame's first parameter) و مو (Shear Modulus)
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
        محاسبه چگالی انرژی پتانسیل کرنش
        F: گرادیان تغییرشکل (ماتریس 3x3)
        """
        # گرادیان جابجایی: H = F - I
        H = F - jnp.eye(3, dtype=F.dtype)

        # تانسور کرنش کوچک متقارن: epsilon = 0.5 * (H + H^T)
        strain = 0.5 * (H + H.T)

        # چگالی انرژی کرنش خطی: 0.5 * lambda * (tr(eps))^2 + mu * tr(eps^2)
        trace_strain = jnp.trace(strain)
        energy_density = (
            0.5 * self.lame_lambda * (trace_strain**2)
            + self.shear_modulus * jnp.sum(strain * strain)
        )
        return energy_density