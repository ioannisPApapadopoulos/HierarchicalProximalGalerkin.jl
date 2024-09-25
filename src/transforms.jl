function plan_legendre_transform(T, szs::Tuple, dims::Tuple=(1,))
    F = NDimsPlan(plan_cheb2leg(T, szs[dims[1]]), szs, dims)
    # F = plan_th_cheb2leg!(T, szs, dims)
    JacobiTransformPlan(F, plan_chebyshevtransform(T, szs, dims))
end

function plan_piecewise_legendre_transform(points::AbstractVector{T}, (M,n)::Tuple{Block{1},Int}, dims::Int) where T
    @assert dims == 1
    F = plan_legendre_transform(T, (Int(M), length(points)-1, n), (dims,))
    ApplyPlan(_perm_blockvec, F, (dims,))
end

function plan_piecewise_legendre_transform(points::AbstractVector{T}, (M,)::Tuple{Block{1}}, dims::Int) where T
    @assert dims == 1
    F = plan_legendre_transform(T, (Int(M), length(points)-1), (dims,))
    ApplyPlan(_perm_blockvec, F, (dims,))
end

function plan_piecewise_legendre_transform(points::AbstractVector{T}, (N,M,n)::Tuple{Block{1},Block{1},Int}, dims=ntuple(identity,Val(2))) where T
    @assert dims == 1:2 || dims == ntuple(identity,Val(2))
    Ns = (N,M)
    F = plan_legendre_transform(T, (_interlace_const(length(points)-1, Int.(Ns)...)..., n), _doubledims(dims...))
    ApplyPlan(_perm_blockvec, F, (dims,))
end

function plan_piecewise_legendre_transform(points::AbstractVector{T}, (N,M)::Tuple{Block{1},Block{1}}, dims=ntuple(identity,Val(2))) where T
    @assert dims == 1:2 || dims == ntuple(identity,Val(2))
    Ns = (N,M)
    F = plan_legendre_transform(T, (_interlace_const(length(points)-1, Int.(Ns)...)), _doubledims(dims...))
    ApplyPlan(_perm_blockvec, F, (dims,))
end