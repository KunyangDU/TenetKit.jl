"""
mul!(C, A, B, α, β) -> C
Combined inplace matrix-matrix or matrix-vector multiply-add \$A B α + C β\$.
The result is stored in C by overwriting it. Note that C must not be aliased with either A or B.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of C, A, B.
Nsweep: times of variational calculation (sweep). Default is 2.
"""
function mul!(C::DenseMPO, A::Union{DenseMPO{L₁},SparseMPO{L₁}}, B::Union{DenseMPO{L₂},SparseMPO{L₂}}, α::Number, β::Number, Alg::Algebraalgo; kwargs...) where {L₁,L₂}

    @assert L₁ == L₂
    to = TimerOutput()
    __init_io__()

    C′ = C'

    @timeit to "initialize ABC Env" begin
        EnvAB = Environment([A,B,C′];isdisk=Alg.isdisk)
        EnvC = Environment([C,C′];isdisk=Alg.isdisk)
        initialize!(EnvAB)
        initialize!(EnvC)
    end
    info = Algebrainfo()
    try
        while info.n ≤ Alg.N
            localto = TimerOutput()

            l2rinfo = Algebrasweepinfo(L2R())
            mto = mul!(EnvC,EnvAB,α,β,Alg,l2rinfo)
            merge!(localto,mto)
            merge!(info,l2rinfo)

            r2linfo = Algebrasweepinfo(R2L())
            mto = mul!(EnvC,EnvAB,α,β,Alg,r2linfo)
            merge!(localto,mto)
            merge!(info,r2linfo)

            _merge_io!(localto)
            show(localto;title = "mul!")
            print("\n")
            show(info)
            merge!(to,localto)
            info.err < Alg.tol && break
        end
        return xp!(C′, C), to, info
    finally
        Alg.isdisk && (cleanup!(EnvAB); cleanup!(EnvC); cleanup!(C′))
    end

    # @assert EnvAB.layer[end] == EnvC.layer[end]
end

function mul!(EnvC::Environment{2}, EnvAB::Environment{3}, α::Number, β::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R}; kwargs...)
    localto = TimerOutput()
    L = length(EnvC.layer[1])
    for site in 1:L-1
        localinfo = Algebrasiteinfo()
        x₀ = composite(EnvC.layer[2][site:site+1]...)
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "composite_AB" tAB = contract(EnvAB.envs[site], vcat(map(u -> EnvAB.layer[u][site:site+1],1:2)...)..., EnvAB.envs[site+2])
        @timeit localto "composite_C" tC = contract(EnvC.envs[site], EnvC.layer[1][site:site+1]..., EnvC.envs[site+2])
        @timeit localto "SVD" tl, tc, tr, truncerr = tsvd(axpby!(α, tC, β, tAB); direction=:center,trunc = Alg.trunc)
        localinfo.truncerr = truncerr
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tr = contract(tc,tr) 
        @timeit localto "push right" map([EnvAB,EnvC]) do Env
            N = length(Env.layer)
            Env.layer[N][site:site+1] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site + 1),1:N)
            pushright!(Env)
        end
        x = composite(EnvC.layer[2][site:site+1]...)
        localinfo.err = norm(x-x₀)^2/x2
        merge!(sweepinfo,localinfo)
    end

    return localto
end

function mul!(EnvC::Environment{2}, EnvAB::Environment{3}, α::Number, β::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L}; kwargs...)
    localto = TimerOutput()
    L = length(EnvC.layer[1])
    for site in L:-1:2
        localinfo = Algebrasiteinfo()
        x₀ = composite(EnvC.layer[2][site-1:site]...)
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "composite_AB" tAB = contract(EnvAB.envs[site-1], vcat(map(u -> EnvAB.layer[u][site-1:site],1:2)...)..., EnvAB.envs[site+1])
        @timeit localto "composite_C" tC = contract(EnvC.envs[site-1], EnvC.layer[1][site-1:site]..., EnvC.envs[site+1])
        @timeit localto "SVD" tl, tc, tr, truncerr = tsvd(axpby!(α, tC, β, tAB); direction=:center,trunc = Alg.trunc)
        localinfo.truncerr = truncerr
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tl = contract(tl,tc) 
        @timeit localto "push left" map([EnvAB,EnvC]) do Env
            N = length(Env.layer)
            Env.layer[N][site-1:site] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site - 1),1:N)
            pushleft!(Env)
        end
        x = composite(EnvC.layer[2][site-1:site]...)
        localinfo.err = norm(x-x₀)^2/x2
        merge!(sweepinfo,localinfo)
    end

    return localto
end

function mul!(EnvC::Environment{2}, EnvAB::Environment{3}, α::Number, β::Number, Alg::Algebraalgo{SingleSite,alg}, sweepinfo::Algebrasweepinfo{L2R}; kwargs...) where alg
    localto = TimerOutput()
    L = length(EnvC.layer[1])
    for site in 1:L-1
        localinfo = Algebrasiteinfo()
        x₀ = composite((EnvC.layer[2][site:site+1])...)
        @assert (x2 = norm(x₀)^2) ≠ 0
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(L2R())
            @timeit localto "CBE_C" cbetoC = CBE!(EnvC, CBEalgo(Alg.alg,DA(),2), cbeinfo)
            @timeit localto "CBE_AB" cbetoAB = CBE!(EnvAB, CBEalgo(Alg.alg,DDA(),3), cbeinfo)
            tLC₀,tRC₀ = EnvC.layer[2][site:site+1]
            tLAB₀,tRAB₀ = EnvAB.layer[3][site:site+1]
            @timeit localto "after-orthogonalize" orthogonalize!(tRAB₀,tRC₀,L2R())
            @timeit localto "direct-sum" tR = _cbedsum(tRAB₀,tRC₀,L2R())
            @timeit localto "splice" tLAB = splice(tLAB₀,tRAB₀,tR,L2R())
            @timeit localto "splice" tLC = splice(tLC₀,tRC₀,tR,L2R())
            EnvC.layer[2][site:site+1] .= tLC,tR 
            EnvAB.layer[3][site:site+1] .= tLAB,tR 
            map([EnvAB,EnvC]) do env
                env.envs[site+1] = pushleft(map(x -> env.layer[x],1:length(env.layer))...,env.envs[site+2],site+1)
            end
            merge!(localinfo,cbeinfo)
            merge!(localto,cbetoC,tree_point = ["CBE_C"])
            merge!(localto,cbetoAB,tree_point = ["CBE_AB"])
        end

        ts = map([EnvC,EnvAB]) do Env 
            @timeit localto "projection" projH = proj1(Env,site)
            action(projH,Env.layer[1][site])
        end

        @timeit localto "orthogonalize" begin
            tl,tr = leftorth(axpby!(α, ts[1], β, ts[2]))
            localinfo.bond = BondInfo(tr)
            tr = contract(tr,EnvC.layer[2][site+1]')
        end
        @timeit localto "push right" map([EnvAB,EnvC]) do Env
            N = length(Env.layer)
            Env.layer[N][site:site+1] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site + 1),1:N)
            pushright!(Env)
        end
        x = composite((EnvC.layer[2][site:site+1])...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
    end

    return localto
end

function mul!(EnvC::Environment{2}, EnvAB::Environment{3}, α::Number, β::Number, Alg::Algebraalgo{SingleSite,alg}, sweepinfo::Algebrasweepinfo{R2L}; kwargs...) where alg
    localto = TimerOutput()
    L = length(EnvC.layer[1])
    for site in L:-1:2
        localinfo = Algebrasiteinfo()
        x₀ = composite((EnvC.layer[2][site-1:site])...)
        @assert (x2 = norm(x₀)^2) ≠ 0
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(R2L())
            @timeit localto "CBE_C" cbetoC = CBE!(EnvC, CBEalgo(Alg.alg,DA(),2), cbeinfo)
            @timeit localto "CBE_AB" cbetoAB = CBE!(EnvAB, CBEalgo(Alg.alg,DDA(),3), cbeinfo)
            tLC₀,tRC₀ = EnvC.layer[2][site-1:site]
            tLAB₀,tRAB₀ = EnvAB.layer[3][site-1:site]
            @timeit localto "after-orthogonalize" orthogonalize!(tLAB₀,tLC₀,R2L())
            @timeit localto "direct-sum" tL = _cbedsum(tLAB₀,tLC₀,R2L())
            @timeit localto "splice" tRAB = splice(tLAB₀,tRAB₀,tL,R2L())
            @timeit localto "splice" tRC = splice(tLC₀,tRC₀,tL,R2L())
            EnvC.layer[2][site-1:site] .= tL,tRAB 
            EnvAB.layer[3][site-1:site] .= tL,tRC 
            map([EnvAB,EnvC]) do env
                env.envs[site] = pushright(map(x -> env.layer[x],1:length(env.layer))...,env.envs[site-1],site-1)
            end
            merge!(localinfo,cbeinfo)
            merge!(localto,cbetoC,tree_point = ["CBE_C"])
            merge!(localto,cbetoAB,tree_point = ["CBE_AB"])
        end
        ts = map([EnvC,EnvAB]) do Env 
            @timeit localto "projection" projH = proj1(Env,site)
            action(projH,Env.layer[1][site])
        end
        @timeit localto "orthogonalize" begin
            tl,tr = rightorth(axpby!(α, ts[1], β, ts[2]))
            localinfo.bond = BondInfo(tl)
            tl = contract(EnvC.layer[2][site-1]',tl)
        end
        @timeit localto "push left" map([EnvAB,EnvC]) do Env
            N = length(Env.layer)
            Env.layer[N][site-1:site] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site - 1),1:N)
            pushleft!(Env)
        end
        x = composite((EnvC.layer[2][site-1:site])...)
        localinfo.err = norm(x-x₀)^2/x2
        merge!(sweepinfo,localinfo)
    end

    return localto
end

function mul!(C::DenseMPO, A::Union{DenseMPO,SparseMPO}, B::Union{DenseMPO,SparseMPO}; kwargs...)
    trunc = get(kwargs,:trunc,notrunc())
    N  = get(kwargs,:N,3)
    tol = get(kwargs,:tol,1e-12)
    algo = get(kwargs,:algo,Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,N,tol))
    return mul!(C,A,B,0,1,algo;kwargs...)
end

