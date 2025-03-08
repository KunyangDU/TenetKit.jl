function CBE!(env::Environment{3},csite::Int64,D::Int64,method::Symbol = :rsvd;kwargs...)
    if method == :rsvd
        λ = get(kwargs, :λ, 1.5)
        ϵ = rsvd!(env,csite,λ,D)
    end
    return ϵ
end

function rsvd!(env::Environment{3},csite::Int64,λ::Number,D::Int64)
    site = env.center[1]
    @assert csite in [site + 1, site - 1]
    if csite == site + 1
        A,Λ = leftorth(env.layer[1].ts[site])
        B = env.layer[1].ts[csite]
        
        EnvL = env.envs[site]
        EnvR = env.envs[csite + 1]

        Lorth = orthogonalize!(env,A,EnvL,site)
        Rorth = orthogonalize!(env,B,EnvR,csite)
        Ω = randntensor(randn,B,λ,:right)
        splice!(Lorth,Λ)
        Q,~ = leftorth(contract(Lorth,splice(Rorth,Ω)))
        Ω = Q'
        splice!(Lorth,Ω)
        obj = contract(Lorth,Rorth)
        ~,Q,ϵ = tsvd(obj;direction = :left,trunc=truncdim(round(Int64,D*λ)))
        orthogonalize!(Q,B,:right)
        Q = dsum(Q,B,:right)

        env.layer[1].ts[csite] = Q
        env.layer[3].ts[csite] = Q'

        env.envs[csite] = pushleft(map(x -> env.layer[x],1:3)...,env.envs[csite+1],csite)
    else
        Λ,B = rightorth(env.layer[1].ts[site])
        A = env.layer[1].ts[csite]
        
        EnvL = env.envs[csite]
        EnvR = env.envs[site + 1]

        Lorth = orthogonalize!(env,A,EnvL,csite)
        Rorth = orthogonalize!(env,B,EnvR,site)

        Ω = randntensor(randn,A,λ,:left)
        splice!(Rorth,Λ)
        ~,Q = rightorth(contract(splice(Lorth,Ω),Rorth))
        Ω = Q'
        splice!(Rorth,Ω)
        obj = contract(Lorth,Rorth)
        Q,~,ϵ = tsvd(obj;direction = :right,trunc=truncdim(round(Int64,D*λ)))
        orthogonalize!(Q,A,:left)
        Q = dsum(Q,A,:left)

        env.layer[1].ts[csite] = Q
        env.layer[3].ts[csite] = Q'

        env.envs[csite + 1] = pushright(map(x -> env.layer[x],1:3)...,env.envs[csite],csite)
    end
    
    return ϵ
end

function randntensor(func,A::MPSTensor{3}, λ::Number,direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if λ != 1
        @assert 0 ≤ λ ≤ 2
        if direction == :left
            tmp = MPSTensor(func,cdm,fuse(cdm))
            tmp,~ = tsvd(MPSTensor(catdomain(tmp.A,tmp.A));direction = :right,trunc = truncdim(round(Int64,λ * dims(dm)[1])))
        elseif direction == :right
            tmp = MPSTensor(func,fuse(cdm[2] ⊗ dm),cdm[2]' ⊗ dm)
            ~,tmp = tsvd(MPSTensor(permute(catcodomain(tmp.A,tmp.A),(1,2),(3,)));direction = :left,trunc = truncdim(round(Int64,λ * dims(cdm)[1])))
        end
    end
    normalize!(tmp)
    return tmp'
end

function randntensor(func,A::DenseMPOTensor{4}, λ::Number,direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if λ != 1
        @assert 0 ≤ λ ≤ 2
        if direction == :left
            tmp = DenseMPOTensor(func,cdm,(fuse(cdm)⊕dm[1])⊗dm[2])
            #tmp.A = permute(tmp.A,(1,2,4),(3,))
            #tmp,~ = tsvd(DenseMPOTensor(permute(catdomain(tmp.A,tmp.A),(1,2),(4,3)));direction = :right,trunc = truncdim(round(Int64,λ * dims(dm)[1])))
        elseif direction == :right
            tmp = DenseMPOTensor(func,cdm[1]⊗(fuse(dm)⊕cdm[2]),dm)
            #tmp.A = permute(tmp.A,(2,),(1,3,4))
            #showdomain(permute(catcodomain(tmp.A,tmp.A),(2,1),(3,4)))
            #~,tmp = tsvd(DenseMPOTensor(permute(catcodomain(tmp.A,tmp.A),(2,1),(3,4)));direction = :left,trunc = truncdim(round(Int64,λ * dims(cdm)[2])))
        end
    end
    normalize!(tmp)
    return tmp'
end

#= function randntensor(func,A::DenseMPOTensor{4}, λ::Number,direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if direction == :left
        tmp = AdjointMPOTensor(func,ℂ^(round(Int64,λ * dims(dm)[1])) ⊗ dm[2],cdm)
    elseif direction == :right
        tmp = AdjointMPOTensor(func,dm,cdm[1] ⊗ ℂ^(round(Int64,λ * dims(cdm)[2])))
    end
    normalize!(tmp)
    return tmp
end =#

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
        ~,Q = rightorth(catcodomain(map(x -> permute(x.A,(2,),(1,3,4)),(Q,A))...))
        Q = DenseMPOTensor(permute(Q,(2,1),(3,4)))
    elseif direction == :left 
        Q,~ = leftorth(catdomain(map(x -> permute(x.A,(1,2,4),(3,)),(Q,A))...))
        Q = DenseMPOTensor(permute(Q,(1,2),(4,3)))
    end
    return Q
end

