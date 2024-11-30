#######
## Proximal Galerkin solver
#######
function pg_hierarchical_solve(PG::Union{<:ObstacleProblem2D{T},BCsObstacleProblem2D{T},ObstacleProblem{T},AdaptiveObstacleProblem{T}, GradientBounds2D{T}}, 
        αs::AbstractVector{T}; 
        initial_guess=(), its_max::Integer=10, pf_its_max::Integer=2,
        backtracking::Bool=true, matrixfree::Bool=false, 
        return_w::Bool=false, show_trace::Bool=true, 
        β::T=0.0, gmres_baseline_tol::T=1e-4, gmres_abstol::T=0.0, restart::Int=200, 
        c_1::T=1e-4, Md=[],tolerance::T=1e-10) where T

    # nu, npsi = ( (PG.p+1) * PG.Nh - 1)^2, (PG.p * PG.Nh)^2
    nu, npsi = size(PG.A,1), size(PG.B,2)
    u, ψ = initial_guess == () ? (zeros(nu), zeros(npsi)) : initial_guess
    w = zeros(npsi)
    gmres_iters = 0
    newton_iters = 0
    ls = BackTracking(c_1=c_1)
    # tics = [0.0;0.0;0.0]
    n_nls = 0
    for α in αs 
        n_nls += 1
        show_trace && print("Considering α=$α.\n")

        # TOL = 1e-3*α # as used for thermoforming
        # gmres_baseline_tol = gmres_baseline_tol/α # as used for thermoforming
        TOL = tolerance
        ls_α = 1.0

        res_u, res_ψ = matrixfree_residual(PG, u, ψ, w, α)
        normres = norm([res_u;res_ψ])   
        show_trace && print("Iteration 0, residual norm: $normres.\n")

        its_max = α == αs[1] ? its_max : pf_its_max
        for iter in 1:its_max
            # res_u, res_ψ = residual(PG, u, ψ, w, α)
            if normres < TOL
                break
            end
            if ls_α < 10000*eps()
                show_trace && print("Linesearch stepsize below eps(), terminating loop. \n")
                break
            end
            if matrixfree
                ((du,dψ), iters)= prec_matrixfree_solve(PG,-res_u,-res_ψ, ψ, α, β=β, gmres_baseline_tol=gmres_baseline_tol, gmres_abstol=gmres_abstol, restart=restart, show_trace=show_trace)
                # ((du,dψ), tic)= mon_prec_solve(PG,-res_u,-res_ψ, ψ, α,w, bandw=bandw, verbosity=1,show_trace=show_trace)    
                # ((du,dψ), iters, tic)= prec_solve2(PG,-res_u,-res_ψ, ψ, α, bandw=bandw, verbosity=1,show_trace=show_trace)    
                
                gmres_iters+=iters
                # tics += tic
            else
                (du,dψ)= assembly_solve(PG,-res_u,-res_ψ, ψ, α, β=β)
                gmres_iters = 0
            end
            newton_iters += 1
            u_ = copy(u)
            (u,ψ,ls_α) = backtracking == true ? pg_linesearch(PG,ls,u,ψ,w,α,du,dψ) : (u+du, ψ+dψ, 1.0)

            res_u, res_ψ = matrixfree_residual(PG, u, ψ, w, α)
            normres = norm([res_u;res_ψ])    

            if !isempty(Md)
                d = u - u_
                cauchy_norm = sqrt(d' * (PG.A + Md) * d)
                show_trace && print("Iteration $iter, stepsize: $ls_α, residual norm: $normres, cauchy norm: $cauchy_norm.\n")
            else
                show_trace && print("Iteration $iter, stepsize: $ls_α, residual norm: $normres.\n")
            end
        end
        if n_nls < lastindex(αs)
            w = copy(ψ)
        end
    end
    if PG isa BCsObstacleProblem2D
        uc = zeros(PG.n2du)
        uc[PG.bcs_idx] = PG.bcs_vals
        uc[PG.free_idx] = u
        u = uc
    end
    if return_w
        return u, ψ, w, (newton_iters, gmres_iters)
    else
        return u, ψ, (newton_iters, gmres_iters)
    end
end