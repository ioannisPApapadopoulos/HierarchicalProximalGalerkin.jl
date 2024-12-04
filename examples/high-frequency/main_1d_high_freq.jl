using MKL
using HierarchicalProximalGalerkin
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles
using SpecialFunctions

"""
Section 7.1: Section 7.1: Solve a 1D obstacle problem with an oscillatory right-hand side.

Here we use the hpG solver with sparse LU factorization and preconditioned GMRES. We consider
    (i)   PDAS, h-uniform, p=1
    (ii)  PDAS, h-adaptive, p=1    
    (ii)  h-adaptive, p-uniform
    (iii) hp-adaptive
    (iv)  h-uniform, p=2,4
    (v)   hp-uniform refinement
    (vi)  p-uniform, h=0.05

"""

include("1d_high_freq_hik.jl")
include("1d_high_freq.jl")
include("1d_high_freq_plots.jl")