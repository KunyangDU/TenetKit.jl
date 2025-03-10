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
        Ω = _cbetensor(randn,B,D_f,:right)
        splice!(Lorth,Λ)
        Q,~ = leftorth(contract(Lorth,splice(Rorth,Ω)))
        Ω = Q'
        splice!(Lorth,Ω)
        obj = contract(Lorth,Rorth)
        orthogonalize!(obj,B,:right)
        ~,Q,ϵ = tsvd(obj;direction = :left,trunc = truncdim(D_f - D_i) & truncbelow(1e-6))
        orthogonalize!(Q,B,:right)
        Q = _cbedsum(Q,B,:right)

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

        Ω = _cbetensor(randn,A,D_f,:left)
        splice!(Rorth,Λ)
        ~,Q = rightorth(contract(splice(Lorth,Ω),Rorth))
        Ω = Q'
        splice!(Rorth,Ω)
        obj = contract(Lorth,Rorth)
        orthogonalize!(obj,A,:left)
        Q,~,ϵ = tsvd(obj;direction = :right,trunc = truncdim(D_f - D_i) & truncbelow(1e-6))
        orthogonalize!(Q,A,:left)
        Q = _cbedsum(Q,A,:left)

        env.layer[1].ts[csite] = Q
        env.layer[3].ts[csite] = Q'

        env.envs[csite + 1] = pushright(map(x -> env.layer[x],1:3)...,env.envs[csite],csite)
    end

    return ϵ
end

#= ===================================== =#

function CBE!(env::Environment{3}, Alg::DMRGalgo, alg::CBEalgo{sch}, info::CBEinfo;kwargs...) where sch
    if sch == randSVD
        return randSVD!(env,Alg,alg,info)
    elseif sch == fullSVD
        return fullSVD!(env,Alg,alg)
    end
end

function randSVD!(env::Environment{3},Alg::DMRGalgo,alg::CBEalgo,info::CBEinfo{L2R})
    @assert (site = env.center[1]) == env.center[2]
    csite = site + 1
    D_f = ceil(Int64,Alg.D*alg.λ)

    A,Λ = leftorth(env.layer[1].ts[site])
    B = env.layer[1].ts[csite]

    D_i = dims(A)[2][1]
    D_i == D_f && return 0
    
    EnvL = env.envs[site]
    EnvR = env.envs[csite + 1]

    Lorth = orthogonalize!(env,A,EnvL,site)
    Rorth = orthogonalize!(env,B,EnvR,csite)
    Ω = _cbetensor(randn,B,D_f,:right)
    splice!(Lorth,Λ)
    Q,~ = leftorth(contract(Lorth,splice(Rorth,Ω)))
    Ω = Q'
    splice!(Lorth,Ω)
    obj = contract(Lorth,Rorth)
    orthogonalize!(obj,B,:right)
    ~,Q,info.ϵ = tsvd(obj;direction = :left,trunc = truncdim(D_f - D_i) & truncbelow(Alg.ϵ))
    orthogonalize!(Q,B,:right)
    Q = _cbedsum(Q,B,:right)

    env.layer[1].ts[csite] = Q
    env.layer[3].ts[csite] = Q'

    env.envs[csite] = pushleft(map(x -> env.layer[x],1:3)...,env.envs[csite+1],csite)
    return info
end

function randSVD!(env::Environment{3},Alg::DMRGalgo,alg::CBEalgo,info::CBEinfo{R2L})
    @assert (site = env.center[1]) == env.center[2]
    csite = site - 1
    D_f = ceil(Int64,Alg.D*alg.λ)

    Λ,B = rightorth(env.layer[1].ts[site])
    A = env.layer[1].ts[csite]

    D_i = dims(A)[2][1]
    D_i == D_f && return 0
    
    EnvL = env.envs[csite]
    EnvR = env.envs[site + 1]

    Lorth = orthogonalize!(env,A,EnvL,csite)
    Rorth = orthogonalize!(env,B,EnvR,site)

    Ω = _cbetensor(randn,A,D_f,:left)
    splice!(Rorth,Λ)
    ~,Q = rightorth(contract(splice(Lorth,Ω),Rorth))
    Ω = Q'
    splice!(Rorth,Ω)
    obj = contract(Lorth,Rorth)
    orthogonalize!(obj,A,:left)
    Q,~,info.ϵ = tsvd(obj;direction = :right,trunc = truncdim(D_f - D_i) & truncbelow(Alg.ϵ))
    orthogonalize!(Q,A,:left)
    Q = _cbedsum(Q,A,:left)

    env.layer[1].ts[csite] = Q
    env.layer[3].ts[csite] = Q'

    env.envs[csite + 1] = pushright(map(x -> env.layer[x],1:3)...,env.envs[csite],csite)
    return info
end
