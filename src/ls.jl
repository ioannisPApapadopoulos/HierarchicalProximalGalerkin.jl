function assembly_solve(PG::Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}, <:ObstacleProblem2D{T}, <:BCsObstacleProblem2D{T}, GradientBounds2D{T}}, 
        u::AbstractVector{T}, v::AbstractVector{T}, ψ::AbstractVector{T}, α::Number;β::Number=1e-10) where T
    A, B = PG.A, PG.B
    M = PG.M

    if !(PG isa AdaptiveObstacleProblem{<:T})
        if PG.ψ ≠ ψ
            PG.D .= assemble_D(PG,ψ)
            PG.ψ .= ψ
        end
        D = sparse(PG.D)
    else
        D = assemble_D(PG,ψ)
    end

    X = [α*A B;-B' D+β*M]
    luX = MatrixFactorizations.lu(X)
    xy = ldiv!(luX, copy([u;v]))

    x = xy[1:size(A,1)]
    y = xy[size(A,1)+1:end]
    (x, y)
end

function assembly_solve(PG:: GradientBounds2D{T}, 
    u::AbstractVector{T}, v::AbstractVector{T}, ψ::AbstractVector{T}, α::Number;β::Number=1e-10) where T
    A, B = PG.A, PG.B
    E = PG.E
    # E = blockdiag(E,E)

    if PG.ψ ≠ ψ
        PG.D .= assemble_D(PG,ψ)
        PG.ψ .= ψ
    end
    D = sparse(PG.D)

    X = [α*A B;-B' D+β*E]
    luX = MatrixFactorizations.lu(X)
    xy = ldiv!(luX, copy([u;v]))

    x = xy[1:size(A,1)]
    y = xy[size(A,1)+1:end]
    (x, y)
end

function prec_matrixfree_solve(PG::Union{<:ObstacleProblem2D{T},<:BCsObstacleProblem2D{T},<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}},
                    u::AbstractVector{T}, v::AbstractVector{T}, ψ::AbstractVector{T}, α::Number;
                    β::T=1e-9,
                    gmres_baseline_tol::T=1e-4, gmres_abstol::T=0.0,
                    show_trace::Bool=true, restart::Int=200) where T
    B = PG.B
    chol_A = PG.chol_A
    E, M = PG.E, PG.M
    n = size(B,2)

    Bt = sparse(B')

    b = Vector(v + Bt * ldiv!(chol_A, copy(u)) / α)

    if PG.ψ ≠ ψ
        PG.D .= assemble_D(PG,ψ)
        PG.ψ .= ψ
    end

    Db = PG.D


    Sf(x) = Db*x + β.*(M*x) + (Bt * ldiv!(chol_A, B*x)) ./ α
    S = LinearMap(Sf, n; ismutating=false)

    Sp = Db + β.*M +  (1e-8 * Diagonal(ones(size(E,1))) + E) ./α
    lu_Sp = MatrixFactorizations.lu(Sp)

    if show_trace
        y, info = IterativeSolvers.gmres(S, b, Pr=lu_Sp, log=true, restart=restart, maxiter=restart, reltol=gmres_baseline_tol, abstol=gmres_abstol, orth_meth=ClassicalGramSchmidt())
        print("GMRES Its: $(info.iters).\n")
        iters = info.iters
        iters ≥ restart && print("WARNING! GMRES iterations: $iters > restart tolerance: $restart.\n")
    else
        y = IterativeSolvers.gmres(S, b, Pr=lu_Sp, restart=restart, maxiter=restart, reltol=gmres_baseline_tol, abstol=gmres_abstol, orth_meth=ClassicalGramSchmidt())
        iters = 0
    end

    x = ldiv!(chol_A, Vector(u - B*y)) ./ α
    (x, y), iters
end

function prec_matrixfree_solve(PG::GradientBounds2D{T},
            u::AbstractVector{T}, v::AbstractVector{T}, ψ::AbstractVector{T}, α::Number;
            β::T=1e-9,
            gmres_baseline_tol::T=1e-4, gmres_abstol::T=0.0,
            show_trace::Bool=true, restart::Int=200) where T

    B = PG.B
    chol_A = PG.chol_A
    E, M = PG.E, PG.M
    n = size(B,2)
    # M = blockdiag(M,M)
    # E = blockdiag(E,E)

    Bt = sparse(B')

    b = Vector(v + Bt * ldiv!(chol_A, copy(u)) / α)

    if PG.ψ ≠ ψ
        PG.D .= assemble_D(PG,ψ)
        PG.ψ .= ψ
    end

    Db = PG.D

    Sf(x) = Db*x + β.*(E*x) + (Bt * ldiv!(chol_A, B*x) ./ α ) 
    S = LinearMap(Sf, n; ismutating=false)

    Sp = Db + β.*E
    lu_Sp = MatrixFactorizations.lu(Sp)

    if show_trace
        y, info = IterativeSolvers.gmres(S, b, Pr=lu_Sp, log=true, restart=restart, maxiter=restart, reltol=gmres_baseline_tol, abstol=gmres_abstol, orth_meth=ClassicalGramSchmidt())
        print("GMRES Its: $(info.iters).\n")
        iters = info.iters
        iters ≥ restart && print("WARNING! GMRES iterations: $iters > restart tolerance: $restart.\n")
    else
        y = IterativeSolvers.gmres(S, b, Pr=lu_Sp, restart=restart, maxiter=restart, reltol=gmres_baseline_tol, abstol=gmres_abstol, orth_meth=ClassicalGramSchmidt())
        iters = 0
    end
    x = ldiv!(chol_A, Vector(u - B*y)) ./ α
    (x, y), iters
end