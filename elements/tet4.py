import jax
import jax.numpy as jnp
try:
    from .base_element import BaseElement
except ImportError:
    from base_element import BaseElement

# تضمین دقت 64 بیتی برای محاسبات بدون خطای گردکردن
jax.config.update("jax_enable_x64", True)


class Tet4(BaseElement):
    """المان خطی ۴ گرهی چهاروجهی با یک نقطه انتگرال‌گیری در مرکز هندسی"""

    n_nodes = 4

    def get_quadrature_data(self):
        """
        محاسبه گرادیان فیزیکی dN/dX و وزن حجم به صورت آرایه‌های استاتیک JAX
        خروجی:
            dN_dX_all: با ابعاد (1, 4, 3) -> (تعداد نقاط گوس، تعداد گره‌ها، بعد فضا)
            weights_all: با ابعاد (1,) شامل حجم المان (det(J) / 6.0)
        """
        # مشتقات توابع شکل استاندارد خطی نسبت به مختصات طبیعی: (4, 3)
        dN_dxi = jnp.array(
            [
                [-1.0, -1.0, -1.0],
                [ 1.0,  0.0,  0.0],
                [ 0.0,  1.0,  0.0],
                [ 0.0,  0.0,  1.0],
            ],
            dtype=self.coordinates.dtype,
        )

        # ماتریس ژاکوبی نگاشت مرجع به فضای فیزیکی: J = X^T @ dN_dxi (ابعاد 3x3)
        jacobian = self.coordinates.T @ dN_dxi
        det_j = jnp.linalg.det(jacobian)

        # مشتق توابع شکل نسبت به مختصات واقعی: dN_dX = dN_dxi @ J^-1 (ابعاد 4x3)
        inv_j = jnp.linalg.inv(jacobian)
        dN_dX = dN_dxi @ inv_j

        # وزن انتگرال‌گیری ۱ نقطه‌ای در چهاروجهی برابر است با حجم آن: V = det(J) / 6.0
        volume = det_j / 6.0

        # افزودن بعد اول (n_quad_points = 1) برای هماهنگی کامل با اینترفیس BaseElement
        dN_dX_all = jnp.expand_dims(dN_dX, axis=0)  # ابعاد: (1, 4, 3)
        weights_all = jnp.array([volume], dtype=self.coordinates.dtype)  # ابعاد: (1,)

        return dN_dX_all, weights_all

    def _energy_density(self, grad_u):
        """تشکیل گرادیان تغییرشکل F و محاسبه چگالی انرژی در ماده"""
        F = jnp.eye(3, dtype=grad_u.dtype) + grad_u
        return self.material.strain_energy(F)