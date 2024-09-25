function assembly_solve(PG::Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}, <:ObstacleProblem2D{T}, <:BCsObstacleProblem2D{T}}, 
        u::AbstractVector{T}, v::AbstractVector{T}, ψ::AbstractVector{T}, α::Number;β::Number=1e-10) where T
    A, B = PG.A, PG.B
    M = PG.M

    if PG.ψ ≠ ψ
        PG.D .= assemble_D(PG,ψ)
        PG.ψ .= ψ
    end
    D = sparse(PG.D)

    X = [α*A B;-B' D+β*M]
    luX = MatrixFactorizations.lu(X)
    xy = ldiv!(luX, copy([u;v]))

    x = xy[1:size(A,1)]
    y = xy[size(A,1)+1:end]
    (x, y)
end

function prec_matrixfree_solve(PG::Union{<:ObstacleProblem2D{T},<:BCsObstacleProblem2D{T},<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}},
                    u::AbstractVector{T}, v::AbstractVector{T}, ψ::AbstractVector{T}, α::Number;
                    β::T=1e-9,
                    gmres_baseline_tol::T=1e-4,
                    show_trace::Bool=true) where T
    B = PG.B
    chol_A = PG.chol_A
    E, M = PG.E, PG.M
    n = size(B,2)

    b = Vector(v + B' * ldiv!(chol_A, copy(u)) / α)

    if PG.ψ ≠ ψ
        PG.D .= assemble_D(PG,ψ)
        PG.ψ .= ψ
    end

    Db = PG.D


    Sf(x) = Vector(  Db*x + β*M*x + B' * ldiv!(chol_A, Vector(B*x)) / α  )
    # Sf(x) = Vector(  apply_D(PG, Vector(x), Vector(ψ)) + β*M*x + B' * ldiv!(chol_A, Vector(B*x)) / α  )
    S = LinearMap(Sf, n; ismutating=false)

    Sp = Db + β*M + E/α
    lu_Sp = MatrixFactorizations.lu(Sp)

    if show_trace
        y, info = IterativeSolvers.gmres(S, b, Pr=lu_Sp, log=true, restart=n, reltol=gmres_baseline_tol/α)
        print("GMRES Its: $(info.iters).\n")
        iters = info.iters
    else
        y = IterativeSolvers.gmres(S, b, Pr=lu_Sp, restart=n, reltol=gmres_baseline_tol/α)
        iters = 0
    end
    
    # tic3 = @elapsed y = IterativeSolvers.gmres(S, b, Pr=lu_Sp, restart=n)#, log=true, restart=n)#, reltol=1e-3/α)

    x = ldiv!(chol_A, Vector(u - B*y)) ./ α
    # ((x, y), iters, [tic1;tic2;tic3])
    (x, y), iters
end