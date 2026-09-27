import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class NeoHookean3D:
    """مدل رفتاری هایپرالاستیک نئوهوکین تراکم‌پذیر در حالت ۳ بعدی"""

    def __init__(self, youngs_modulus: float, poisson_ratio: float):
        self.youngs_modulus = jnp.float64(youngs_modulus)
        self.poisson_ratio = jnp.float64(poisson_ratio)

        # محاسبه ثوابت لامه: مو (مدول برشی) و لامبدا
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
        محاسبه چگالی انرژی پتانسیل کرنش
        ورودی:
            F: گرادیان تغییرشکل (ماتریس 3x3)
        خروجی:
            اسکالر انرژی چگالی کرنش
        """
        # تانسور کوشی-گرین راست: C = F^T @ F
        C = F.T @ F

        # ناوردای اول کرنش: I_C = tr(C)
        I_C = jnp.trace(C)

        # نسبت تغییر حجم المان: J = det(F)
        J = jnp.linalg.det(F)

        # لگاریتم تغییر حجم
        ln_J = jnp.log(J)

        # انرژی کرنش نئوهوکین تراکم‌پذیر
        psi = (
            0.5 * self.shear_modulus * (I_C - 3.0 - 2.0 * ln_J)
            + 0.5 * self.lame_lambda * (ln_J**2)
        )
        return psi