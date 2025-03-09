function CBE!(env::Environment{3},csite::Int64,trunc::TruncationScheme,method::Symbol = :rsvd;kwargs...)
    if method == :rsvd
        λ = get(kwargs, :λ, 1.2)
        ϵ = rsvd!(env,csite,λ,trunc)
    end
    return ϵ
end

function rsvd!(env::Environment{3},csite::Int64,λ::Number,trunc::TruncationScheme)
    site = env.center[1]
    @assert csite in [site + 1, site - 1]
    D_f = ceil(Int64,_getdim(trunc)*λ)
    if csite == site + 1
        A,Λ = leftorth(env.layer[1].ts[site])
        B = env.layer[1].ts[csite]

        D_i = dims(A)[2][1]
        D_i == D_f && return 0
        
        EnvL = env.envs[site]
        EnvR = env.envs[csite + 1]

        Lorth = orthogonalize!(env,A,EnvL,site)
        Rorth = orthogonalize!(env,B,EnvR,csite)
        Ω = randntensor(randn,B,D_f,:right)
        # Ω = orthogonalize!(Ω',B,:right)'
        splice!(Lorth,Λ)
        Q,~ = leftorth(contract(Lorth,splice(Rorth,Ω)))
        Ω = Q'
        splice!(Lorth,Ω)
        obj = contract(Lorth,Rorth)
        orthogonalize!(obj,B,:right)
        ~,Q,ϵ = tsvd(obj;direction = :left,trunc = truncdim(D_f - D_i) & truncbelow(1e-6))
        orthogonalize!(Q,B,:right)
        Q = dsum(Q,B,:right)

        env.layer[1].ts[csite] = Q
        env.layer[3].ts[csite] = Q'

        env.envs[csite] = pushleft(map(x -> env.layer[x],1:3)...,env.envs[csite+1],csite)
    else
        Λ,B = rightorth(env.layer[1].ts[site])
        A = env.layer[1].ts[csite]

        D_i = dims(A)[2][1]
        D_i == D_f && return 0
        
        EnvL = env.envs[csite]
        EnvR = env.envs[site + 1]

        Lorth = orthogonalize!(env,A,EnvL,csite)
        Rorth = orthogonalize!(env,B,EnvR,site)

        Ω = randntensor(randn,A,D_f,:left)
        Ω = orthogonalize!(Ω',A,:left)'
        splice!(Rorth,Λ)
        ~,Q = rightorth(contract(splice(Lorth,Ω),Rorth))
        Ω = Q'
        splice!(Rorth,Ω)
        obj = contract(Lorth,Rorth)
        orthogonalize!(obj,A,:left)
        Q,~,ϵ = tsvd(obj;direction = :right,trunc = truncdim(D_f - D_i) & truncbelow(1e-6))
        orthogonalize!(Q,A,:left)
        Q = dsum(Q,A,:left)

        env.layer[1].ts[csite] = Q
        env.layer[3].ts[csite] = Q'

        env.envs[csite + 1] = pushright(map(x -> env.layer[x],1:3)...,env.envs[csite],csite)
    end

    return ϵ
end

_expanddim(::ComplexSpace,D::Int64) = ℂ^D

function _expanddim(S::GradedSpace,D::Int64)
    ratio = D / dim(S)
    for (c,d) in S.dims
        S.dims[c] = ceil(Int64,d*ratio)
    end
    return S
end

function randntensor(func,A::MPSTensor{3}, D_f::Int64, direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if direction == :left
        tmp = MPSTensor(func,cdm,_expanddim(fuse(cdm),D_f))
    elseif direction == :right
        tmp = MPSTensor(func,_expanddim(fuse((cdm[2] ⊗ dm)),D_f) ⊗ cdm[2],dm)
    end
    normalize!(tmp)
    return tmp'
end

function randntensor(func,A::DenseMPOTensor{4}, D_f::Int64,direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if direction == :left
        tmp = DenseMPOTensor(func,cdm,(_expanddim(fuse(cdm⊗dm[2]),D_f) )⊗dm[2])
    elseif direction == :right
        tmp = DenseMPOTensor(func,cdm[1]⊗(_expanddim(fuse(dm⊗cdm[1]),D_f)),dm)
    end
    normalize!(tmp)
    return tmp'
end

function dsum(Q::MPSTensor{3},A::MPSTensor{3},direction::Symbol)
    if direction == :right
        ~,Q = rightorth(catcodomain(map(x -> permute(x.A,(1,),(2,3)),(Q,A))...))
        Q = MPSTensor(permute(Q,(1,2),(3,)))
    elseif direction == :left 
        Q,~ = leftorth(catdomain(Q.A,A.A))
        Q = MPSTensor(Q)
    end
    return Q
end

function dsum(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::Symbol)
    if direction == :right
        # Q = catcodomain(map(x -> permute(x.A,(2,),(1,3,4)),(Q,A))...)
        # Q = DenseMPOTensor(permute(Q,(2,1),(3,4)))
        Q = catcodomain(map(x -> permute(x.A,(2,),(1,3,4)),(Q,A))...)
        ~,Q = rightorth(DenseMPOTensor(permute(Q,(2,1),(3,4))))
    elseif direction == :left 
        # Q = catdomain(map(x -> permute(x.A,(1,2,4),(3,)),(Q,A))...)
        # Q = DenseMPOTensor(permute(Q,(1,2),(4,3)))
        Q = catdomain(map(x -> permute(x.A,(1,2,4),(3,)),(Q,A))...)
        Q,~ = leftorth(DenseMPOTensor(permute(Q,(1,2),(4,3))))
    end
    return Q
end

