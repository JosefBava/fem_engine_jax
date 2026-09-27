"""Small, extensible JAX finite-element engine."""

from .elements import Hex8, Tet4
from .materials import Hooke3D, NeoHookean3D
from .mesh import Mesh, BoundaryConditions
from .solvers import NewtonRaphson

__all__ = ["Hex8", "Tet4", "Hooke3D", "NeoHookean3D", "Mesh", "BoundaryConditions", "NewtonRaphson"]