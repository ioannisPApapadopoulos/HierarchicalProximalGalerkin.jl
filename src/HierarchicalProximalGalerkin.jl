module HierarchicalProximalGalerkin

using BlockBandedMatrices, ClassicalOrthogonalPolynomials, PiecewiseOrthogonalPolynomials, LinearAlgebra,
        IterativeSolvers, LinearMaps, MatrixFactorizations, SparseArrays, LineSearches, BlockArrays,
        ExtendableSparse # Preconditioners,
import ClassicalOrthogonalPolynomials: grid, pad, oneto, ClosedInterval, jacobimatrix, mortar, Clenshaw, recurrencecoefficients, _p0, band, Ones, JacobiTransformPlan
import MatrixFactorizations: reversecholcopy, reversecholesky!
import PiecewiseOrthogonalPolynomials: blocksize, BandedMatrix, reversecholesky, BBBArrowheadMatrix, BlockVec, plan_grid_transform, ApplyPlan,
         _perm_blockvec, _inv_perm_blockvec, _doubledims, _interlace_const
import LinearAlgebra: ldiv!, \, *, mul!
import BlockBandedMatrices: _BandedBlockBandedMatrix
import LinearAlgebra: \, ldiv!
import FastTransforms: ArrayPlan, NDimsPlan, plan_cheb2leg, plan_th_cheb2leg!, plan_chebyshevtransform

export ContinuousPolynomial, DirichletPolynomial, weaklaplacian, grammatrix, Block, BlockVec, grid, plan_transform, plan_grid_transform, expand, pad, blocksize,oneto,
        ObstacleProblem,    
        apply_D, _apply_D, apply_Db, apply_D_alias, matrixfree_solve, assembly_solve, matrixfree_residual, prec_matrixfree_solve, assemble_D_alias,
        newton, ldiv!,
        assemble_D, assemble_Db, assemble_banded_D,
        _BandedBlockBandedMatrix, sparse,
        jacobimatrix,
        l2_norm_psi, l2_norm_u, h1_norm_u, l2_norm_u_fast, h1_norm_u_fast,
        pg_hierarchical_solve,
        ldiv!,
        find_intersect, pg_linesearch, pg_plotD, pg_plotP, evaluate_u,
        ObstacleProblem2D,
        AdaptiveObstacleProblem, active_indices_A, active_indices_M, pad_u, pad_ψ, cellwise_decomposition,cellwise_interlace,
        MeshAdaptivity, analyticity_coeffs, error_estimates, h_refine, hp_refine, uniform_refine, hp_uniform_refine,
        MeshAdaptivity2D, zero_pad,
        HIKSolver, HIKSolver2D, solve, evaluate, hik_gridap_solve,
        Thermoforming2D,
        normH1, normL2, evaluate2D,
        BCsObstacleProblem2D,
        GradientBounds2D

include("lvpp/proximalgalerkin.jl")
include("lvpp/proximalgalerkin_2d.jl")
include("lvpp/proximalgalerkin_2d_bcs.jl")
include("lvpp/proximalgalerkin_adaptive.jl")
include("lvpp/gradientbounds.jl")
include("matrixfree.jl")
include("assembly.jl")
include("ls.jl")
include("nls.jl")
include("misc/linesearch.jl")
include("misc/misc.jl")
include("misc/HIKsolver.jl")
include("misc/thermoforming.jl")
include("adaptivity/hp_adaptivity.jl")
include("adaptivity/hp_adaptivity_2d.jl")
include("transforms.jl")



function l2_norm_u(D::DirichletPolynomial, u::AbstractVector{T}, v::AbstractVector{T}, M::AbstractMatrix{T}, cutoff::Integer=20_000) where T
        d = pad(u[1:cutoff]-v[1:cutoff], axes(D,2))
        d, sqrt((M * d)' * d)
end
function h1_norm_u(d::AbstractVector{T}, A::AbstractMatrix{T}, l2::T) where T
        sqrt((A * d)' * d + l2^2)
end
function l2_norm_psi(P::ContinuousPolynomial{0}, u::AbstractVector{T}, v::AbstractVector{T}, Mp::AbstractMatrix{T}, cutoff::Integer=20_000) where T
        d = pad(u[1:cutoff]-v[1:cutoff], axes(P,2))
        sqrt((Mp * d)' * d)
end

function l2_norm_u_fast(u::AbstractVector{T}, v::AbstractVector{T}, M::AbstractMatrix{T}) where T
        d = u - v
        d, sqrt((M * d)' * d)
end
function h1_norm_u_fast(d::AbstractVector{T}, A::AbstractMatrix{T}, l2::T) where T
        sqrt((A * d)' * d + l2^2)
end

function normH1(PG::Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}}, u) where T
        A = PG.A
        M = grammatrix(PG.Dp)[Block.(1:PG.p+1), Block.(1:PG.p+1)]
        sqrt(u' * A * u + u' * M * u)
end

function normH1(PG::ObstacleProblem2D{T}, u) where T
        A = PG.A
        M1 = sparse(view(grammatrix(PG.Dp), Block.(1:PG.p+1), Block.(1:PG.p+1)))
        M = Symmetric(kron(M1, M1))
        sqrt(u' * A * u + u' * M * u)
end

function normL2(PG::Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}}, u) where T
        M = grammatrix(PG.Dp)[Block.(1:PG.p+1), Block.(1:PG.p+1)]
        sqrt(u' * M * u)
end

function normL2(PG::ObstacleProblem2D{T}, u) where T
        M1 = sparse(view(grammatrix(PG.Dp), Block.(1:PG.p+1), Block.(1:PG.p+1)))
        M = Symmetric(kron(M1, M1))
        sqrt(u' * M * u)
end

ldiv!(p::SparseArrays.CHOLMOD.Factor, b) = copyto!(b, p \ b)
function ldiv!(y, p::SparseArrays.CHOLMOD.Factor{T}, x) where T
    y = p \ x
end


function jacobimatrix(P::ContinuousPolynomial{0}, p::Integer)
    T = eltype(P)
    r = P.points
    Nh = length(r)-1

    intervals = ClosedInterval{T}[P.points[j]..P.points[j+1] for j in 1:Nh]
    J = jacobimatrix.(legendre.(intervals))
    Jb  = BBBArrowheadMatrix(zeros(0,0), (), (), J)[Block.(2:p+1), Block.(2:p+1)]
    data = vcat(Jb.data[2,:]', Jb.data[5,:]', Jb.data[8,:]')
    _BandedBlockBandedMatrix(data, repeat([Nh], p), repeat([Nh], p), (1,1), (0,0))
end

end # module HierarchicalProximalGalerkin
