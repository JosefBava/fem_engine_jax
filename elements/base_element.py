from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp

# الزامی برای محاسبات دقیق المان محدود
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
        خروجی باید آرایه‌های استاتیک JAX باشند (نه ژنراتور پایتونی):
        dN_dX_all: با ابعاد (n_quad_points, n_nodes, 3)
        weights_all: با ابعاد (n_quad_points,) شامل وزن گوس ضربدر det(J)
        """
        raise NotImplementedError

    def compute_energy(self, u_flat, dN_dX_all, weights_all):
        """محاسبه انرژی کل پتانسیل کرنش المان"""
        u_nodes = u_flat.reshape((self.n_nodes, 3))

        def point_energy(dN_dX, w_detJ):
            # محاسبه گرادیان جابجایی: H_ij = d(u_i) / d(X_j)
            grad_u = u_nodes.T @ dN_dX  # ماتریس 3x3
            F = jnp.eye(3) + grad_u     # گرادیان تغییرشکل F
            
            # دریافت انرژی از ماده (مثلاً نئوهوکین)
            psi = self.material.strain_energy(F)
            return psi * w_detJ

        # جمع روی تمام نقاط گوس به صورت برداری
        energies = jax.vmap(point_energy)(dN_dX_all, weights_all)
        return jnp.sum(energies)

    def get_tangent_and_residual(self, u_e):
        """محاسبه ماتریس سختی مماس و بردار پسماند"""
        u_flat = jnp.asarray(u_e, dtype=jnp.float64).reshape(-1)
        dN_dX_all, weights_all = self.get_quadrature_data()

        # تعریف تابع انرژی فقط نسبت به جابجایی
        energy_fn = lambda u: self.compute_energy(u, dN_dX_all, weights_all)

        # گرادیان انرژی = نیروی داخلی المان (F_int)
        # در روش نیوتن پسماند برابر است با: R = F_ext - F_int
        f_int = jax.grad(energy_fn)(u_flat)
        residual = -f_int

        # هسین انرژی = ماتریس سختی تانژانت المان (Ke)
        tangent = jax.hessian(energy_fn)(u_flat)

        return tangent, residual