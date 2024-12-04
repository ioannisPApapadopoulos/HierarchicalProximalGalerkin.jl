using MKL
using HierarchicalProximalGalerkin
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles
using SpecialFunctions

"""
This example requires the package ```SpecialFunctions.jl```

Section 7.2: Solve a 2D obstacle problem with a bessel-type oscillatory obstacle.

Here we use the hpG solver with sparse LU factorization and preconditioned GMRES. We consider
    (i)   p-uniform refinement
    (ii)  h-uniform refinement with p=2, 3
    (iii) hp-uniform refinement

We also solve via the primal-dual active set strategy with h-uniform refinement, p=1.

"""

include("bessel_obstacle_hik.jl")
include("bessel_obstacle_2d.jl")
include("bessel_obstacle_plots.jl")
include("bessel_obstacle_gmres.jl")