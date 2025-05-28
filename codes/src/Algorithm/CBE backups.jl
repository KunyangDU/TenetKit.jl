
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
#     D_i ≥ D_f && return localto

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
#     D_i ≥ D_f && return localto
    
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

# function CBE!(env::Environment{3}, alg::CBEalgo{sch}, info::CBEinfo{R2L};kwargs...) where sch
#     tL,tR,localto = CBE(env,alg,info)
#     isnothing(tL) && isnothing(tR) && return localto
#     site = env.center[1]

#     env.layer[1].ts[site-1] = tL
#     env.layer[3].ts[site-1] = tL'
#     env.layer[1].ts[site] = tR
#     env.layer[3].ts[site] = tR'

#     env.envs[site] = pushright(map(x -> env.layer[x],1:3)...,env.envs[site-1],site-1)
#     return localto
# end


# function randSVD(env::Environment{3}, alg::CBEalgo,info::CBEinfo{L2R})
#     @assert (site = env.center[1]) == env.center[2]
    
#     localto = TimerOutput()

#     tL₀,tR₀ = env.layer[1].ts[site:site+1]
#     EnvL = env.envs[site]
#     EnvR = env.envs[site + 2]
#     hl,hr = env.layer[2].ts[site:site+1]
    
#     D_i = dims(tL₀)[2][1]
#     D_f = ceil(Int64,alg.D*alg.λ)
#     D_i ≥ D_f && return nothing,nothing,localto

#     Ω = _cbetensor(randn,tR₀,D_f,:right)

#     @timeit localto "leftorth" tL,Λ = leftorth(tL₀)

#     @timeit localto "left orthogonalize" Lorth = orthogonalize!(hl,tL,EnvL)
#     @timeit localto "right orthogonalize" Rorth = orthogonalize!(hr,tR₀,EnvR)

#     @timeit localto "splice Λ" splice!(Lorth,Λ)
#     @timeit localto "contract" Q = contract(Lorth,splice(Rorth,Ω))
#     @timeit localto "leftorth" Q,~ = leftorth(Q)
#     @timeit localto "splice Q'" splice!(Lorth,Q')
#     @timeit localto "contract" obj = contract(Lorth,Rorth)
#     # @timeit localto "pre-orthogonalize" orthogonalize!(obj,tR₀,:right)
#     @timeit localto "SVD" ~,Q,info.err = tsvd(obj;direction = :left,trunc = truncdim(D_f - D_i) & truncbelow(alg.ϵ))
#     @timeit localto "after-orthogonalize" orthogonalize!(Q,tR₀,:right)

#     @timeit localto "direct-sum" tR = _cbedsum(Q,tR₀,:right)
#     @timeit localto "splice" tL = splice(tL₀,tR₀,tR,L2R())

#     return tL,tR,localto
# end

# function randSVD(env::Environment{3},alg::CBEalgo,info::CBEinfo{R2L})
#     @assert (site = env.center[1]) == env.center[2]
    
#     localto = TimerOutput()

#     tL₀,tR₀ = env.layer[1].ts[site-1:site]
#     EnvL = env.envs[site - 1]
#     EnvR = env.envs[site + 1]
#     hl,hr = env.layer[2].ts[site-1:site]

#     D_i = dims(tL₀)[2][1]
#     D_f = ceil(Int64,alg.D*alg.λ)
#     D_i ≥ D_f && return nothing,nothing,localto
#     Ω = _cbetensor(randn,tL₀,D_f,:left)

#     @timeit localto "rightorth" Λ,tR = rightorth(tR₀)
    
#     @timeit localto "left orthogonalize" Lorth = orthogonalize!(hl,tL₀,EnvL)
#     @timeit localto "right orthogonalize" Rorth = orthogonalize!(hr,tR,EnvR)

#     @timeit localto "splice Λ" splice!(Rorth,Λ)
#     @timeit localto "contract" Q = contract(splice(Lorth,Ω),Rorth)
#     @timeit localto "rightorth" ~,Q = rightorth(Q)
#     @timeit localto "splice Q'" splice!(Rorth,Q')
#     @timeit localto "contract" obj = contract(Lorth,Rorth)
#     # @timeit localto "pre-orthogonalize" orthogonalize!(obj,tL₀,:left)
#     @timeit localto "SVD" Q,~,info.err = tsvd(obj;direction = :right,trunc = truncdim(D_f - D_i) & truncbelow(alg.ϵ))
#     @timeit localto "after-orthogonalize" orthogonalize!(Q,tL₀,:left)

#     @timeit localto "direct-sum" tL = _cbedsum(Q,tL₀,:left)
#     @timeit localto "splice" tR = splice(tL₀,tR₀,tL,R2L())

#     return tL,tR,localto
# end


# function CBE(env::Environment{N}, alg::CBEalgo{sch}, info::CBEinfo;kwargs...) where {N,sch}
#     if sch == randSVD
#         return randSVD(env,alg,info)
#     elseif sch == fullSVD
#         return fullSVD(env,alg)
#     end
# end