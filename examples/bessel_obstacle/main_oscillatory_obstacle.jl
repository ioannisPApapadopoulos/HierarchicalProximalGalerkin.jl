using MKL
using HierarchicalProximalGalerkin
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles
using SpecialFunctions

include("examples/bessel_obstacle/bessel_obstacle_hik.jl")
include("examples/bessel_obstacle/bessel_obstacle_2d.jl")
include("examples/bessel_obstacle/bessel_obstacle_plots.jl")
include("examples/bessel_obstacle/bessel_obstacle_gmres.jl")