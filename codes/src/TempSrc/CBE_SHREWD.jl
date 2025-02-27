function orthogonalize!(env::Environment{3},Λ::MPSTensor{2},B::MPSTensor{3},EnvR::SparseRightEnvironmentTensor,osite::Int64)
    w,w2 = env.layer[2].D[osite]
    EnvRorth = Vector(undef,w)
    EnvRorth .= nothing

    for i in 1:w, j in 1:w2
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(Λ,B,Hij,EnvR.A[j])
        if isnothing(EnvRorth[i])
            EnvRorth[i] = tmp - contract(tmp,B)
        else
            EnvRorth[i] += tmp - contract(tmp,B)
        end
    end

    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
end

function orthogonalize!(env::Environment{3},Λ::MPSTensor{2},A::MPSTensor{3},EnvL::SparseLeftEnvironmentTensor,osite::Int64)
    w1,w = env.layer[2].D[osite]
    EnvLorth = Vector(undef,w)
    EnvLorth .= nothing

    for i in 1:w1, j in 1:w
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(Λ,A,Hij,EnvL.A[i])
        if isnothing(EnvLorth[j])
            EnvLorth[j] = tmp - contract(tmp,A)
        else
            EnvLorth[j] += tmp - contract(tmp,A)
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
end

function contract(Λ::MPSTensor{2},B::MPSTensor{3},H::DenseMPOTensor{2},EnvR::RightEnvironmentTensor{2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ Λ.A[-1,3] * B.A[3,2,1] * H.A[-2,2] * EnvR.A[1,-3])
end

function contract(EnvR::RightCompositeEnvironmentTensor{1,3}, B::MPSTensor{3})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,2,1] * B'.A[1,3,2] * B.A[3,-2,-3])
end

function contract(Λ::MPSTensor{2},A::MPSTensor{3},H::DenseMPOTensor{2},EnvL::LeftEnvironmentTensor{2})
    return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,1] * A.A[1,2,3] * H.A[-2,2] * Λ.A[3,-3])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2,3}, A::MPSTensor{3})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3])
end

function Base.:-(A::RightCompositeEnvironmentTensor{N₁, R₁}, B::RightCompositeEnvironmentTensor{N₂, R₂}) where {N₁,N₂,R₁,R₂}
    @assert N₁ == N₂ && R₁ == R₂
    return RightCompositeEnvironmentTensor(A.A - B.A)
end

function Base.:-(A::LeftCompositeEnvironmentTensor{N₁, R₁}, B::LeftCompositeEnvironmentTensor{N₂, R₂}) where {N₁,N₂,R₁,R₂}
    @assert N₁ == N₂ && R₁ == R₂
    return LeftCompositeEnvironmentTensor(A.A - B.A)
end

function TensorKit.tsvd(A::RightCompositeEnvironmentTensor{N,R};kwargs...) where {N,R}
    U,S,V,ϵ = tsvd(A.A,Tuple(1:R-N-1),Tuple(R-N:R);kwargs...)
    d = sqrt(@tensor S[1,2] * S'[2,1])
    !iszero(d) && (ϵ /= d )
    return MPSTensor(U*S),RightCompositeEnvironmentTensor(permute(V,Tuple(1:R-N),Tuple(R-N+1:R))),ϵ
end

function TensorKit.tsvd(A::LeftCompositeEnvironmentTensor{N,R};kwargs...) where {N,R}
    U,S,V,ϵ = tsvd(A.A,Tuple(1:N,),Tuple(N+1:R);kwargs...)
    d = sqrt(@tensor S[1,2] * S'[2,1])
    !iszero(d) && (ϵ /= d )
    return LeftCompositeEnvironmentTensor(permute(U,Tuple(1:N),Tuple(N+1:R))),MPSTensor(S*V),ϵ
end

function TensorKit.tsvd(A::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor};kwargs...)
    Rs = []
    Ls = []
    ϵs = []
    for i in 1:A.D
        L,R,ϵ = tsvd(A.A[i];kwargs...)
        push!(Ls,L)
        push!(Rs,R)
        push!(ϵs,ϵ)
    end
    return Ls,Rs,ϵs
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, Λ::MPSTensor{2})
    return LeftCompositeEnvironmentTensor(EnvL.A*Λ.A)
end

function contract(EnvL::RightCompositeEnvironmentTensor{1, 3}, Λ::MPSTensor{2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3])
end

function splice!(Envorth::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor},Λ::MPSTensor{2})
    for i in 1:Envorth.D
        Envorth.A[i] = contract(Envorth.A[i],Λ)
    end
end

function bisect(envorth::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor},D::Int64)
    envs = []
    ϵs = []
    for env in envorth.A
        tmp,ϵ0 = bisect(env,D)
        push!(envs,tmp)
        push!(ϵs,ϵ0)
    end
    return envs,ϵs
end

function bisect(env::LeftCompositeEnvironmentTensor{N,R},D::Int64) where {N,R}
    U,S,V,ϵ = tsvd(env.A,Tuple(1:N),Tuple(N+1:R);trunc = truncdim(D))
    d = sqrt(@tensor S[1,2] * S'[2,1])
    !iszero(d) && (ϵ /= d )
    return MPSTensor(U),ϵ
end

function bisect(env::RightCompositeEnvironmentTensor{N,R},D::Int64) where {N,R}
    U,S,V,ϵ = tsvd(env.A,Tuple(1:R-N-1),Tuple(R-N:R);trunc = truncdim(D))
    d = sqrt(@tensor S[1,2] * S'[2,1])
    !iszero(d) && (ϵ /= d )
    return MPSTensor(permute(V,(1,2),(3,))),ϵ
end

function preselect(env::Environment{3},csite::Int64)
    site = env.center[1]
    if csite == site + 1
        B = env.layer[1].ts[csite]
        DR0 = dims(B)[1][1]
        A,Λ = tsvd(env.layer[1].ts[site];direction = :right)
        Lorth = orthogonalize!(env,Λ,A,env.envs[site],site)
        ~,Λs,ϵs1 = tsvd(Lorth;trunc = truncdim(DR0))
        Λ = sum(Λs)
        Rorth = orthogonalize!(env,Λ,B,env.envs[csite+1],csite)
        finals,ϵs2 = bisect(Rorth,round(Int64,DR0/Rorth.D))
        final = trivialdensify(convert(Vector{MPSTensor{3}},finals),:left)

        if  final*B' != 0
            @tensor final.A[-1,-2;-3] = final.A[-1,-2,-3] - final.A[-1,1,2] * B'.A[2,3,1] * B.A[3,-2,-3]
        end

        @assert abs(final*B') < 1e-10

        ϵ = max(ϵs1...,ϵs2...)
    elseif csite == site - 1
        A = env.layer[1].ts[csite]
        Λ,B = tsvd(env.layer[1].ts[site];direction = :left)

        Rorth = orthogonalize!(env,Λ,B,env.envs[site+1],site)
        Λs,~,ϵs1 = tsvd(Rorth;trunc = truncdim(dims(A)[2][1]))
        Λ = sum(Λs)

        Lorth = orthogonalize!(env,Λ,A,env.envs[csite],csite)
        finals,ϵs2 = bisect(Lorth,Int64(dims(A)[2][1]/Lorth.D))

        final = trivialdensify(convert(Vector{MPSTensor{3}},finals),:right)
        if  final*A' != 0
            @tensor final.A[-1,-2;-3] = final.A[-1,-2,-3] - final.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3]
        end

        @assert abs(final*A') < 1e-10

        ϵ = max(ϵs1...,ϵs2...)
    else
        @error "site incorrect"
    end
    return final,ϵ
end

function CBE!(env::Environment{3},csite::Int64)
    @assert (site = env.center[1]) == env.center[2]
    pres = shrewd(env,csite)
    if csite == site + 1
        
    elseif csite == site - 1
        
    end
    
end

function trivialdensify(Λs::Vector{MPSTensor{2}},direction::Symbol)
    L = length(Λs)
    cdom,dom = dims(Λs[1])
    if direction == :left
        return MPSTensor(TensorMap(cat([Λs[i].A.data for i in 1:L]...;dims=3),ℂ^(L*cdom[1]),ℂ^dom[1]))    
    elseif direction == :right
        return MPSTensor(TensorMap(cat([Λs[i].A.data for i in 1:L]...;dims=3),ℂ^(cdom[1]),ℂ^(L*dom[1])))    
    end
end

function trivialdensify(ts::Vector{MPSTensor{3}},direction::Symbol)
    L = length(ts)
    cdom,dom = dims(ts[1])
    if direction == :left
        tmp = TensorMap(cat([permute(ts[i].A,(2,3),(1,)).data for i in 1:L]...;dims=3),ℂ^(cdom[2]) ⊗ (ℂ^(dom[1]))' ,(ℂ^(L*cdom[1]))')
        return MPSTensor(permute(tmp,(3,1),(2,)))
    elseif direction == :right
        return MPSTensor(TensorMap(cat([ts[i].A.data for i in 1:L]...;dims=3),ℂ^(cdom[1])⊗ℂ^(cdom[2]),ℂ^(L*dom[1])))
    end
end