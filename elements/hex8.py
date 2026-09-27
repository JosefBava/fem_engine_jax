import jax
import jax.numpy as jnp

try:
    from .base_element import BaseElement
except ImportError:
    from base_element import BaseElement

# فعال‌سازی الزامی دقت 64 بیتی
jax.config.update("jax_enable_x64", True)


class Hex8(BaseElement):
    """المان ۸ گرهی سه‌بعدی با انتگرال‌گیری گاوس ۲×۲×۲"""

    n_nodes = 8

    # مختصات گره‌های مرجع (مستقر روی float64)
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
        محاسبه گرادیان فیزیکی dN/dX و وزن‌ها به صورت یکپارچه (تنسوری)
        خروجی:
            dN_dX_all: آرایه‌ای با ابعاد (8, 8, 3) -> (تعداد نقاط گوس, تعداد گره‌ها, بعد فضا)
            weights_all: آرایه‌ای با ابعاد (8,) حاوی دترمینان ژاکوبی هر نقطه
        """
        point = 1.0 / jnp.sqrt(3.0)
        # تعریف مختصات ۸ نقطه انتگرال‌گیری گاوس
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
            # محاسبه مشتقات توابع شکل نسبت به مختصات طبیعی المان
            dN_dxi = 0.125 * jnp.stack(
                (
                    self._signs[:, 0] * (1.0 + self._signs[:, 1] * xi[1]) * (1.0 + self._signs[:, 2] * xi[2]),
                    self._signs[:, 1] * (1.0 + self._signs[:, 0] * xi[0]) * (1.0 + self._signs[:, 2] * xi[2]),
                    self._signs[:, 2] * (1.0 + self._signs[:, 0] * xi[0]) * (1.0 + self._signs[:, 1] * xi[1]),
                ),
                axis=1,
            )  # ابعاد: (8, 3)

            # ماتریس ژاکوبی: J = X^T @ dN_dxi
            jacobian = self.coordinates.T @ dN_dxi
            det_j = jnp.linalg.det(jacobian)

            # مشتق توابع شکل نسبت به مختصات فیزیکی: dN_dX = dN_dxi @ J^-1
            dN_dX = dN_dxi @ jnp.linalg.inv(jacobian)
            return dN_dX, det_j

        # ارزیابی خودکار روی تمام ۸ نقطه گاوس به صورت هم‌زمان
        dN_dX_all, weights_all = jax.vmap(compute_single_gp)(gauss_points)
        return dN_dX_all, weights_all

    def _energy_density(self, grad_u):
        """دریافت گرادیان جابجایی و ارسال گرادیان تغییرشکل F به مدل ماده"""
        F = jnp.eye(3, dtype=grad_u.dtype) + grad_u
        return self.material.strain_energy(F)