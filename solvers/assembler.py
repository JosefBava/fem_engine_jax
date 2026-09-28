import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def assemble_system(mesh, elements_list, displacement, boundary_conditions=None):
    """
    Assemble the global stiffness matrix and residual vector for a finite element problem.
    """
    n_dofs = mesh.n_dofs
    dtype = displacement.dtype

    stiffness = jnp.zeros((n_dofs, n_dofs), dtype=dtype)
    residual = jnp.zeros(n_dofs, dtype=dtype)

    # 1 extract local stiffness and residual for each element and assemble into global system
    for index, elem in enumerate(elements_list):
        # Get the degrees of freedom (DOFs) for the current element and extract the corresponding displacements from the global displacement vector.
        dofs = mesh.get_element_dofs(index)
        u_e = displacement[dofs]

        # solve for local stiffness matrix and residual vector for the element
        local_k, local_r = elem.get_tangent_and_residual(u_e)

        # extract local stiffness and residual for each element and assemble into global system
        stiffness = stiffness.at[jnp.ix_(dofs, dofs)].add(local_k)
        residual = residual.at[dofs].add(local_r)

    # 2. (Neumann)
    if boundary_conditions is not None:
        f_ext = boundary_conditions.get_external_force(n_dofs)
        residual = residual + f_ext

        # 3. (Dirichlet) 
        stiffness, residual = boundary_conditions.apply_to_nonlinear_step(
            stiffness, residual, displacement
        )

    return stiffness, residual