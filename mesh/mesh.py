from dataclasses import dataclass
import jax
import jax.numpy as jnp

# تنظیم پیش‌فرض دقت ۶۴ بیتی
jax.config.update("jax_enable_x64", True)


@dataclass
class Mesh:
    """کلاس نگه‌داری داده‌های شبکه المان محدود سازگار با JAX"""
    coordinates: jnp.ndarray          # ابعاد: (n_nodes, 3)
    connectivities: jnp.ndarray       # ابعاد: (n_elements, n_nodes_per_elem)

    def __post_init__(self):
        # ذخیره‌سازی مختصات با دقت بالا و اتصالات با اندیس عدد صحیح
        self.coordinates = jnp.asarray(self.coordinates, dtype=jnp.float64)
        self.connectivities = jnp.asarray(self.connectivities, dtype=jnp.int32)

    @property
    def n_nodes(self) -> int:
        """تعداد کل گره‌های مش"""
        return self.coordinates.shape[0]

    @property
    def n_elements(self) -> int:
        """تعداد کل المان‌ها"""
        return self.connectivities.shape[0]

    @property
    def n_dofs(self) -> int:
        """تعداد کل درجات آزادی (۳ درجه آزادی به ازای هر گره در ۳ بعد)"""
        return 3 * self.n_nodes

    def element_coordinates(self, index: int) -> jnp.ndarray:
        """دریافت مختصات گره‌های یک المان به صورت منفرد"""
        return self.coordinates[self.connectivities[index]]

    def all_element_coordinates(self) -> jnp.ndarray:
        """
        استخراج هم‌زمان مختصات تمام المان‌ها جهت استفاده مستقیم در jax.vmap
        خروجی: آرایه‌ای با ابعاد (n_elements, n_nodes_per_elem, 3)
        """
        return self.coordinates[self.connectivities]

    def get_element_dofs(self, index: int) -> jnp.ndarray:
        """
        استخراج اندیس درجات آزادی سراسری متناظر با یک المان
        ترتیب DOFها: [u_x1, u_y1, u_z1, u_x2, u_y2, u_z2, ...]
        """
        node_indices = self.connectivities[index]
        # ایجاد درجات آزادی ۳ جهته برای هر گره المان
        dofs = jnp.stack(
            [3 * node_indices, 3 * node_indices + 1, 3 * node_indices + 2],
            axis=-1
        ).reshape(-1)
        return dofs