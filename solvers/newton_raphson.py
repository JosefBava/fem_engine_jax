from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any
import jax
import jax.numpy as jnp

# تضمین دقت محاسباتی ۶۴ بیتی
jax.config.update("jax_enable_x64", True)


@dataclass
class NewtonRaphson:
    """ A class implementing the Newton-Raphson method for solving nonlinear systems of equations in finite element analysis. It iteratively updates the displacement vector until convergence is achieved based on the residual norm and specified tolerance."""
    tolerance: float = 1e-8
    max_iterations: int = 25
    verbose: bool = False

    def solve(
        self,
        residual_and_tangent_fn: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
        initial_displacement: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        K_T * delta_u = R
        u_{k+1} = u_k + delta_u
        """
        displacement = jnp.asarray(initial_displacement, dtype=jnp.float64)
        residual_history = []

        for iteration in range(1, self.max_iterations + 1):
            # residual_and_tangent_fn is a function that takes the current displacement vector and returns the residual vector and tangent stiffness matrix.
            tangent, residual = residual_and_tangent_fn(displacement)
            norm = float(jnp.linalg.norm(residual))
            residual_history.append(norm)

            if self.verbose:
                print(f"Iteration {iteration:02d} | Residual Norm: {norm:.4e}")

            # check for convergence based on the residual norm and specified tolerance. If the norm is less than the tolerance, the method has converged, and we return the current displacement along with convergence information.
            if norm < self.tolerance:
                return displacement, {
                    "converged": True,
                    "iterations": iteration - 1,
                    "residual_norm": norm,
                    "history": residual_history,
                }

            # If the method has not converged, we solve for the displacement increment (delta_u) by solving the linear system defined by the tangent stiffness matrix and the residual vector. We then update the current displacement vector by adding delta_u to it.
            delta_u = jnp.linalg.solve(tangent, residual)
            displacement = displacement + delta_u

        # If the method reaches the maximum number of iterations without converging, we return the last computed displacement along with information indicating that convergence was not achieved, including the last residual norm and the history of residual norms over the iterations.
        last_norm = float(jnp.linalg.norm(residual))
        return displacement, {
            "converged": False,
            "iterations": self.max_iterations,
            "residual_norm": last_norm,
            "history": residual_history,
        }