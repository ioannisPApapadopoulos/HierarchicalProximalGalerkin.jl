struct MeshAdaptivity2D{T}
    PG::Union{<:ObstacleProblem2D{T}, <:BCsObstacleProblem2D}
    α::T
    up::AbstractMatrix{T}
    exp_ψP::AbstractMatrix{T}
    λ::AbstractMatrix{T}
    oc::AbstractMatrix{T}
    fP::AbstractMatrix{T}
    φP::AbstractMatrix{T}
    gP::AbstractMatrix{T}
    Rp::AbstractMatrix{T}
    ΔP::AbstractMatrix{T}
    M::AbstractMatrix{T}
end

function MeshAdaptivity2D(PG::Union{<:ObstacleProblem2D{T},BCsObstacleProblem2D{T}}, α::T, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, f::Function, φ::Function;bcs=[]) where T
    P, p = PG.P, PG.p
    C = PG isa ObstacleProblem2D ? PG.Dp : PG.C

    λ = (w-ψ)/α
    xy, plan_P = plan_grid_transform(PG.P, (Block(p+2),Block(p+2)))
    x,y = first(xy), last(xy)
    nx, px, ny, py = size(plan_P)

    Rp1 = sparse(view(P \ C, Block.(1:p+2), Block.(1:p+1)))
    Rp = kron(Rp1, Rp1)
    up = Rp*u
    up = BlockMatrix(reshape(up, nx*px, ny*py), repeat([px], nx), repeat([py], ny))

    λ = BlockMatrix(reshape(λ, (nx-2)*px, (ny-2)*py), repeat([px], nx-2), repeat([py], ny-2))

    fP = (plan_P * f.(x,reshape(y,1,1,size(y)...)))
    φP = (plan_P * φ.(x,reshape(y,1,1,size(y)...)))

    if PG isa BCsObstacleProblem2D
        @assert bcs isa Function
        function g(x1,x2)
            if x[1,1] < x1 < x[end,end] &&  y[1,1] < x2 < y[end,end]
                return 0.0
            else
                return bcs(x1,x2)
            end
        end
        gP = (plan_P * g.(x,reshape(y,1,1,size(y)...)))
    else
        gP = zeros(1)
    end

    ψx = plan_P \ BlockMatrix(zero_pad(reshape(ψ, (nx-2)*px, (ny-2)*py),(p+2)*PG.Nh), repeat([px], nx), repeat([py], ny))
    ψx[ψx.<-100].=-100
    exp_ψP = plan_P * exp.(-ψx)

    oc = (plan_P * max.(plan_P \ (up - φP), 0))

    L = Legendre()
    x = axes(L,1)
    Dm = sparse((Jacobi(2,2) \ (Derivative(x) * (Derivative(x) * L)))[1:p+2,1:p+2])
    Rm = sparse((L \ Jacobi(2,2))[1:p+2,1:p+2])
    Mm = sparse(Diagonal((L' * L)[1:p+2, 1:p+2]))
    
    RDm = Rm*Dm
    ΔP = kron(RDm, Mm) + kron(Mm, RDm)

    M = grammatrix(PG.P)[Block.(1:p+2),Block.(1:p+2)]


    MeshAdaptivity2D{T}(PG, α, up, exp_ψP, λ, oc, fP, φP, gP, Rp, ΔP, M)
end


function zero_pad(M::AbstractMatrix{T}, n::Integer) where T
    szM = size(M,1)
    @assert szM == size(M,2)
    @assert szM ≤ n

    A = hcat(M, zeros(T, szM, n-szM))
    A = vcat(A, zeros(T, n-szM, n))
    A
end

function error_estimates(MA::MeshAdaptivity2D{T}, pg::Bool=false) where T
    PG = MA.PG
    p, n = PG.p, PG.Nh


    up,λ,oc = MA.up, MA.λ, MA.oc
    λ = zero_pad(λ, (p+2)*n)

    r = PG.P.points

    ΔP, M = MA.ΔP, MA.M

    ϵs = T[]

    fc = zero_pad(BlockMatrix(reshape(PG.f, p*n, p*n), repeat([n], p), repeat([n], p)), (p+2)*n)
    φc = zero_pad(BlockMatrix(reshape(PG.φ, p*n, p*n), repeat([n], p), repeat([n], p)), (p+2)*n)

    fP = MA.fP
    φP = MA.φP


    for K1 in 1:n
        for K2 in 1:n
            fv = fc[K1:n:end, K2:n:end]
            fPv = fP[K1:n:end, K2:n:end]
            λv = λ[K1:n:end, K2:n:end]
            uv = up[K1:n:end, K2:n:end]
            φv = φc[K1:n:end, K2:n:end]
            φPv = φP[K1:n:end, K2:n:end]

            hT = sqrt( (r[K1+1]-r[K1])^2 + (r[K2+1]-r[K2])^2 )
            pT = p
            
            dv = (fv[:] + 4/hT^2*ΔP*uv[:] + λv[:])

            Mv = kron(M[K1:n:end,K1:n:end],M[K2:n:end,K2:n:end])

            if PG isa BCsObstacleProblem2D
                if K1==1 || K1==n || K2==1 || K2==n
                    gPv = MA.gP[K1:n:end, K2:n:end]
                    bd = (gPv - uv)[:]
                    bosc = bd' * Mv * bd
                else
                    bosc = 0.0
                end
            else
                bosc = 0.0
            end

            df = (fPv - fv)[:]
            dφ = (φPv - φv)[:]
            osc = hT^2/pT^2 *(df' * Mv * df) + dφ' * Mv * dφ + bosc
            if pg
                exp_ψPv = exp_ψP[K1:n:end, K2:n:end]
                ev = (uv - φv + exp_ψPv)[:]
                Res = sqrt(MA.α^2 * hT^2/pT^2 * dv' * Mv  * dv + ev' * Mv * ev + osc)
            else
                ov = oc[K1:n:end, K2:n:end]
                Res = sqrt(hT^2/pT^2 * dv' * Mv  * dv + ov[:]' * Mv * ov[:] + abs(λv[:]' * Mv * (φv-uv)[:]) + osc)
            end
            push!(ϵs, Res)
        end
    end
    reshape(ϵs,n,n)

end

function h_refine(MA::MeshAdaptivity2D{T}, ϵs::AbstractMatrix{T}; δ::T=0.1) where T
    is_err = ϵs .> δ*maximum(ϵs)
    # @assert is_err == is_err'
    is_err = LowerTriangular(is_err)

    refine_idx = Integer[]
    for i in 1:size(is_err,1)÷2
        if sum(is_err[i:end-i+1,i]) > 0
            push!(refine_idx, i)
        end
        if sum(is_err[end-i+1,i:end-i+1]) > 0
            push!(refine_idx, size(is_err,1)-i+1)
        end
    end
    refine_idx = sort(refine_idx)
    r = MA.PG.P.points
    sort(r ∪ (r[refine_idx.+1]+r[refine_idx])/2)
end