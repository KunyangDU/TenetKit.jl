
# function CBE!(env::Environment{3}, alg::CBEalgo{sch}, info::CBEinfo;kwargs...) where sch
#     if sch == randSVD
#         return randSVD!(env,alg,info)
#     elseif sch == fullSVD
#         return fullSVD!(env,alg)
#     end
# end

# function randSVD!(env::Environment{3}, alg::CBEalgo,info::CBEinfo{L2R})
#     @assert (site = env.center[1]) == env.center[2]
#     csite = site + 1
#     D_f = ceil(Int64,alg.D*alg.λ)
#     localto = TimerOutput()

#     @timeit localto "leftorth" A,Λ = leftorth(env.layer[1].ts[site])
#     B = env.layer[1].ts[csite]

#     D_i = dims(A)[2][1]
#     D_i == D_f && return localto

#     EnvL = env.envs[site]
#     EnvR = env.envs[csite + 1]

#     @timeit localto "left orthogonalize" Lorth = orthogonalize!(env,A,EnvL,site)
#     @timeit localto "right orthogonalize" Rorth = orthogonalize!(env,B,EnvR,csite)
#     Ω = _cbetensor(randn,B,D_f,:right)
#     @timeit localto "splice Λ" splice!(Lorth,Λ)
#     @timeit localto "contract" Q = contract(Lorth,splice(Rorth,Ω))
#     @timeit localto "leftorth" Q,~ = leftorth(Q)
#     @timeit localto "splice Q'" splice!(Lorth,Q')
#     @timeit localto "contract" obj = contract(Lorth,Rorth)
#     # @timeit localto "pre-orthogonalize" orthogonalize!(obj,B,:right)
#     @timeit localto "SVD" ~,Q,info.err = tsvd(obj;direction = :left,trunc = truncdim(D_f - D_i) & truncbelow(alg.ϵ))
#     @timeit localto "after-orthogonalize" orthogonalize!(Q,B,:right)
#     Q = _cbedsum(Q,B,:right)

#     env.layer[1].ts[csite] = Q
#     env.layer[3].ts[csite] = Q'

#     env.envs[csite] = pushleft(map(x -> env.layer[x],1:3)...,env.envs[csite+1],csite)
#     return localto
# end

# function randSVD!(env::Environment{3},alg::CBEalgo,info::CBEinfo{R2L})
#     @assert (site = env.center[1]) == env.center[2]
#     csite = site - 1
#     D_f = ceil(Int64,alg.D*alg.λ)
#     localto = TimerOutput()

#     @timeit localto "rightorth" Λ,B = rightorth(env.layer[1].ts[site])
#     A = env.layer[1].ts[csite]

#     D_i = dims(A)[2][1]
#     D_i == D_f && return localto
    
#     EnvL = env.envs[csite]
#     EnvR = env.envs[site + 1]

#     @timeit localto "left orthogonalize" Lorth = orthogonalize!(env,A,EnvL,csite)
#     @timeit localto "right orthogonalize" Rorth = orthogonalize!(env,B,EnvR,site)

#     Ω = _cbetensor(randn,A,D_f,:left)
#     @timeit localto "splice Λ" splice!(Rorth,Λ)
#     @timeit localto "contract" Q = contract(splice(Lorth,Ω),Rorth)
#     @timeit localto "rightorth" ~,Q = rightorth(Q)
#     @timeit localto "splice Q'" splice!(Rorth,Q')
#     @timeit localto "contract" obj = contract(Lorth,Rorth)
#     # @timeit localto "pre-orthogonalize" orthogonalize!(obj,A,:left)
#     @timeit localto "SVD" Q,~,info.err = tsvd(obj;direction = :right,trunc = truncdim(D_f - D_i) & truncbelow(alg.ϵ))
#     @timeit localto "after-orthogonalize" orthogonalize!(Q,A,:left)
#     Q = _cbedsum(Q,A,:left)

#     env.layer[1].ts[csite] = Q
#     env.layer[3].ts[csite] = Q'

#     env.envs[csite + 1] = pushright(map(x -> env.layer[x],1:3)...,env.envs[csite],csite)
#     return localto
# end

function CBE!(env::Environment{3}, alg::CBEalgo{sch}, info::CBEinfo{L2R};kwargs...) where sch
    tl,tr,localto = CBE(env,alg,info)
    site = env.center[1]

    env.layer[1].ts[site] = tl
    env.layer[3].ts[site] = tl'
    env.layer[1].ts[site+1] = tr
    env.layer[3].ts[site+1] = tr'

    env.envs[site+1] = pushleft(map(x -> env.layer[x],1:3)...,env.envs[site+2],site+1)
    return localto
end

function CBE!(env::Environment{3}, alg::CBEalgo{sch}, info::CBEinfo{R2L};kwargs...) where sch
    tl,tr,localto = CBE(env,alg,info)
    site = env.center[1]

    env.layer[1].ts[site-1] = tl
    env.layer[3].ts[site-1] = tl'
    env.layer[1].ts[site] = tr
    env.layer[3].ts[site] = tr'

    env.envs[site] = pushright(map(x -> env.layer[x],1:3)...,env.envs[site-1],site-1)
    return localto
end

function CBE(env::Environment{3}, alg::CBEalgo{sch}, info::CBEinfo;kwargs...) where sch
    if sch == randSVD
        return randSVD(env,alg,info)
    elseif sch == fullSVD
        return fullSVD(env,alg)
    end
end


function randSVD(env::Environment{3}, alg::CBEalgo,info::CBEinfo{L2R})
    @assert (site = env.center[1]) == env.center[2]
    csite = site + 1
    D_f = ceil(Int64,alg.D*alg.λ)
    localto = TimerOutput()

    @timeit localto "leftorth" A,Λ = leftorth(env.layer[1].ts[site])
    B = env.layer[1].ts[csite]

    D_i = dims(A)[2][1]
    D_i == D_f && return localto

    EnvL = env.envs[site]
    EnvR = env.envs[csite + 1]

    @timeit localto "left orthogonalize" Lorth = orthogonalize!(env,A,EnvL,site)
    @timeit localto "right orthogonalize" Rorth = orthogonalize!(env,B,EnvR,csite)
    Ω = _cbetensor(randn,B,D_f,:right)
    @timeit localto "splice Λ" splice!(Lorth,Λ)
    @timeit localto "contract" Q = contract(Lorth,splice(Rorth,Ω))
    @timeit localto "leftorth" Q,~ = leftorth(Q)
    @timeit localto "splice Q'" splice!(Lorth,Q')
    @timeit localto "contract" obj = contract(Lorth,Rorth)
    # @timeit localto "pre-orthogonalize" orthogonalize!(obj,B,:right)
    @timeit localto "SVD" ~,Q,info.err = tsvd(obj;direction = :left,trunc = truncdim(D_f - D_i) & truncbelow(alg.ϵ))
    @timeit localto "after-orthogonalize" orthogonalize!(Q,B,:right)

    @timeit localto "direct-sum" tr = _cbedsum(Q,B,:right)
    @timeit localto "splice" tl = splice(env.layer[1].ts[site:site+1]...,tr,L2R())
    
    return tl,tr,localto
end

function randSVD(env::Environment{3},alg::CBEalgo,info::CBEinfo{R2L})
    @assert (site = env.center[1]) == env.center[2]
    csite = site - 1
    D_f = ceil(Int64,alg.D*alg.λ)
    localto = TimerOutput()

    @timeit localto "rightorth" Λ,B = rightorth(env.layer[1].ts[site])
    A = env.layer[1].ts[csite]

    D_i = dims(A)[2][1]
    D_i == D_f && return localto
    
    EnvL = env.envs[csite]
    EnvR = env.envs[site + 1]

    @timeit localto "left orthogonalize" Lorth = orthogonalize!(env,A,EnvL,csite)
    @timeit localto "right orthogonalize" Rorth = orthogonalize!(env,B,EnvR,site)

    Ω = _cbetensor(randn,A,D_f,:left)
    @timeit localto "splice Λ" splice!(Rorth,Λ)
    @timeit localto "contract" Q = contract(splice(Lorth,Ω),Rorth)
    @timeit localto "rightorth" ~,Q = rightorth(Q)
    @timeit localto "splice Q'" splice!(Rorth,Q')
    @timeit localto "contract" obj = contract(Lorth,Rorth)
    # @timeit localto "pre-orthogonalize" orthogonalize!(obj,A,:left)
    @timeit localto "SVD" Q,~,info.err = tsvd(obj;direction = :right,trunc = truncdim(D_f - D_i) & truncbelow(alg.ϵ))
    @timeit localto "after-orthogonalize" orthogonalize!(Q,A,:left)

    @timeit localto "direct-sum" tl = _cbedsum(Q,A,:left)
    @timeit localto "splice" tr = splice(env.layer[1].ts[site-1:site]...,tl,R2L())

    return tl,tr,localto
end
