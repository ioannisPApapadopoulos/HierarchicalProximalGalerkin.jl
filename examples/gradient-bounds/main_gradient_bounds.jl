using MKL
using HierarchicalProximalGalerkin
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles
using SpecialFunctions

"""
Section 7.3: Solve a generalized elastic-plastic torsion problem via the hpG solver.

Here we use the hpG solver with sparse LU factorization and preconditioned GMRES. We consider
    (i)   p-uniform refinement
    (ii)  h-uniform refinement with p=1,2,3
    (iii) hp-uniform refinement

"""

include("gradient_bounds_reference.jl")
include("gradient_bounds.jl")
include("gradient_bounds_plots.jl")
include("gradient_bounds_gmres.jl")