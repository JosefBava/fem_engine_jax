import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def assemble_system(mesh, elements_list, displacement, boundary_conditions=None):
    """
    اسمبلر سراسری سازگار با گراف محاسباتی JAX
    mesh: نمونه کلاس Mesh
    elements_list: لیستی از نمونه‌های از پیش ساخته‌شده المان‌ها
    displacement: بردار جابجایی سراسری فعلی (u)
    boundary_conditions: شیء کلاس BoundaryConditions
    """
    n_dofs = mesh.n_dofs
    dtype = displacement.dtype

    stiffness = jnp.zeros((n_dofs, n_dofs), dtype=dtype)
    residual = jnp.zeros(n_dofs, dtype=dtype)

    # 1. حلقه اسمبلی درجات آزادی المان‌ها
    for index, elem in enumerate(elements_list):
        # استخراج اندیس DOFs المان
        dofs = mesh.get_element_dofs(index)
        u_e = displacement[dofs]

        # دریافت سفتی مماس و پسماند از المان
        local_k, local_r = elem.get_tangent_and_residual(u_e)

        # مونتاژ در ماتریس و بردار سراسری
        stiffness = stiffness.at[jnp.ix_(dofs, dofs)].add(local_k)
        residual = residual.at[dofs].add(local_r)

    # 2. اضافه کردن بار خارجی (Neumann)
    if boundary_conditions is not None:
        f_ext = boundary_conditions.get_external_force(n_dofs)
        residual = residual + f_ext

        # 3. اعمال شرایط مرزی تکیه‌گاهی (Dirichlet)
        stiffness, residual = boundary_conditions.apply_to_nonlinear_step(
            stiffness, residual, displacement
        )

    return stiffness, residual