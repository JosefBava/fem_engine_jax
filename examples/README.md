# FEM benchmark examples

These examples cover both implemented 3D element types and both material models.

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

Each example uses continuation for stable loading and writes these files to `results/`:

- `<name>.png`: undeformed/deformed 3D view colored by displacement magnitude.
- `<name>_convergence.png`: residual norm for every Newton iteration.
- `<name>_iterations.csv`: load step, iteration number, and residual norm.
- `<name>.pvd`: ParaView collection containing only the undeformed and final deformed states.

Open the `.pvd` file in ParaView. Time `0` is the undeformed mesh and time `1` is the converged deformed mesh. The VTU files contain the deformed coordinates plus `Displacement` and `DisplacementMagnitude` point fields. The full load/Newton history remains available in the CSV and convergence PNG.