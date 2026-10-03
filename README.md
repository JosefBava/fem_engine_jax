# FEM benchmark examples




### Hex8 Hooke compression


<img width="896" height="348" alt="hex8_hook_com" src="https://github.com/user-attachments/assets/dbd6af0d-5288-47b1-9633-23c8bc113be3" />

### Hex8 Neo-Hookean shear


<img width="896" height="348" alt="hex8_neoh_shear" src="https://github.com/user-attachments/assets/47cca6cd-fab3-475f-b883-0f1b5282da78" />

### Tet4 Hooke cantilever


<img width="896" height="348" alt="tet4_hook_cant" src="https://github.com/user-attachments/assets/4be4425a-37ea-420d-97f3-ae1149f84126" />

### Tet4 Neo-Hookean tension


<img width="896" height="348" alt="tet4_neoh_ten" src="https://github.com/user-attachments/assets/d0dd280b-871e-40c6-a577-bed67b2da52e" />

These are useful for presentation, validation, and visual comparison of the benchmark response across different element formulations and constitutive laws.


This finite-element example set demonstrates the JAX-based Newton solver for both linear-elastic and hyperelastic 3D solids. The examples compare `Hex8` and `Tet4` element formulations and validate the consistent tangent implementation through convergence plots and load-displacement curves.

| Example | Element | Material | Problem |
| --- | --- | --- | --- |
| `tet4_hooke_cantilever` | Tet4 | Hooke3D | Distributed tip load on a cantilever |
| `hex8_hooke_compression` | Hex8 | Hooke3D | Constrained block in compression |
| `tet4_neo_hookean_tension` | Tet4 | NeoHookean3D | Large-strain uniaxial tension |
| `tet4_neo_hookean_fixed_tension` | Tet4 | NeoHookean3D | Prescribed 0.02 displacement on a unit cube |
| `hex8_neo_hookean_shear` | Hex8 | NeoHookean3D | Prescribed large shear deformation |

Run from `FEM/GITHUB/FEM`:

```powershell
python -m fem_engine.examples.tet4_hooke_cantilever
python -m fem_engine.examples.hex8_hooke_compression
python -m fem_engine.examples.tet4_neo_hookean_tension
python -m fem_engine.examples.tet4_neo_hookean_fixed_tension
python -m fem_engine.examples.hex8_neo_hookean_shear
```

Each example uses continuation loading and writes benchmark outputs to `results/`:

- `<name>.png`: undeformed/deformed 3D view colored by displacement magnitude.
- `<name>_convergence.png`: residual norm as a function of Newton iteration, shown on a logarithmic scale to highlight quadratic convergence behavior.
- `<name>_load_displacement.png`: load factor vs. measured displacement at the loaded surface.
- `<name>_iterations.csv`: load step, iteration number, and residual norm.
- `<name>_load_curve.csv`: summary of load factor and average top-surface displacement per load step.
- `<name>.pvd`: ParaView collection containing the undeformed and final deformed states.

Open the `.pvd` file in ParaView. Time `0` is the undeformed mesh and time `1` is the converged deformed mesh. The VTU files contain the deformed coordinates plus `Displacement` and `DisplacementMagnitude` point fields. The full Newton history remains available in the CSV and convergence PNG. The generated load-displacement curves are useful for comparing the linear Hooke response against the large-strain Neo-Hookean response within the same finite-element framework.



