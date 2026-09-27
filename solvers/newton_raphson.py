from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any
import jax
import jax.numpy as jnp

# تضمین دقت محاسباتی ۶۴ بیتی
jax.config.update("jax_enable_x64", True)


@dataclass
class NewtonRaphson:
    """حل‌کننده غیرخطی نیوتن-رافسون سازگار با JAX"""
    tolerance: float = 1e-8
    max_iterations: int = 25
    verbose: bool = False

    def solve(
        self,
        residual_and_tangent_fn: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
        initial_displacement: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        اجرای حلقه تعادل نیوتن-رافسون: K_T * delta_u = R
        u_{k+1} = u_k + delta_u
        """
        displacement = jnp.asarray(initial_displacement, dtype=jnp.float64)
        residual_history = []

        for iteration in range(1, self.max_iterations + 1):
            # ارزیابی ماتریس مماس و بردار پسماند
            tangent, residual = residual_and_tangent_fn(displacement)
            norm = float(jnp.linalg.norm(residual))
            residual_history.append(norm)

            if self.verbose:
                print(f"Iteration {iteration:02d} | Residual Norm: {norm:.4e}")

            # بررسی شرط همگرایی
            if norm < self.tolerance:
                return displacement, {
                    "converged": True,
                    "iterations": iteration - 1,
                    "residual_norm": norm,
                    "history": residual_history,
                }

            # حل سیستم معادلات خطی مماس برای گام تصحیح
            delta_u = jnp.linalg.solve(tangent, residual)
            displacement = displacement + delta_u

        # در صورت عدم همگرایی در حداکثر تعداد تکرار
        last_norm = float(jnp.linalg.norm(residual))
        return displacement, {
            "converged": False,
            "iterations": self.max_iterations,
            "residual_norm": last_norm,
            "history": residual_history,
        }