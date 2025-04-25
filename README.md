# HierarchicalProximalGalerkin

This repository implement the hierarchical proximal Galerkin solver (hpG) as described in

"Hierarchical proximal Galerkin: a fast $hp$-FEM solver for variational problems with pointwise inequality constraints", Ioannis. P. A. Papadopoulos (2024), https://arxiv.org/abs/2412.13733.

It combines the proximal Galerkin solver of Keith and Surowiec ([paper](https://doi.org/10.1007/s10208-024-09681-8)) with the hierarchical p-FEM basis to achieve very-high-order discretizations of:
    (i)   the obstacle problem,
    (ii)  the generalized elastic-plastic torsion problem, and
    (iii) the thermoforming problem: an obstacle-type quasi-variational inequality.


|Figure|File: examples/|
|:-:|:-:|
|1, 2, 3|[spy-plots.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/spy-plots.jl)|
|4|[high-frequency/main_1d_high_freq.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/high-frequency/main_1d_high_freq.jl)|
|5, 6|[bessel_obstacle/main_oscillatory_obstacle.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/bessel_obstacle/main_oscillatory_obstacle.jl)|
|7|[gradient-bounds/main_gradient_bounds.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/gradient-bounds/main_gradient_bounds.jl)|
|8|[thermoforming.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/thermoforming.jl)|

|Table|File: examples/|
|:-:|:-:|
|1|[bessel_obstacle/main_oscillatory_obstacle.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/bessel_obstacle/main_oscillatory_obstacle.jl)|
|2|[gradient-bounds/main_gradient_bounds.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/gradient-bounds/main_gradient_bounds.jl)|
|3|[thermoforming.jl](https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl/blob/main/examples/thermoforming.jl)|

## Installation

The package is not registered. Please install via

```pkg> add https://github.com/ioannisPApapadopoulos/HierarchicalProximalGalerkin.jl.git```

## Dependencies

This package relies on [PiecewiseOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/PiecewiseOrthogonalPolynomials.jl) v0.5.1 for its implementation of the hierarchical p-FEM basis and [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl) v0.16.5 that allow for fast quadrature.

It also utilizes [SparseArrays.jl](https://github.com/JuliaSparse/SparseArrays.jl), [MatrixFactorizations.jl](https://github.com/JuliaLinearAlgebra/MatrixFactorizations.jl), and [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) for fast linear algebra.


## References

If you use this package please reference:

[1] Ioannis P. A. Papadopoulos. "Hierarchical proximal Galerkin: a fast $hp$-FEM solver for variational problems with pointwise inequality constraints" (2024), https://arxiv.org/abs/2412.13733.

[2] Brendan Keith and Thomas M. Surowiec. "Proximal Galerkin: A structure-preserving finite element method for pointwise bound constraints." Foundations of Computational Mathematics (2024): 1-97. [DOI: 10.1007/s10208-024-09681-8](https://doi.org/10.1007/s10208-024-09681-8) 


## Contact

papadopoulos@wias-berlin.de



