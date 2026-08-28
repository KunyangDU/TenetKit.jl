function mul!(C::Union{DenseMPO{L₁},DenseMPS{L₁}}, A::Union{DenseMPO{L₁},DenseMPS{L₁}}, B::Union{DenseMPO{L₂},SparseMPO{L₂},AdjointMPO{L₂},RefMPO{L₂}}, α::Number, Alg::Algebraalgo; kwargs...) where {L₁,L₂}

    @assert L₁ == L₂
    to = TimerOutput()
    __init_io__()
    C′ = C'
    @timeit to "initialize ABC Env" begin
        EnvAB = Environment([A,B,C′];isdisk=Alg.isdisk)
        initialize!(EnvAB;kwargs...)
    end

    info = Algebrainfo()
    try
        while info.n ≤ Alg.N
            info.err = 0
            localto = TimerOutput()

            l2rinfo = Algebrasweepinfo(L2R())
            mto = mul!(EnvAB,α,Alg,l2rinfo)

            show(mto;title = ">>> mul! - $(info.n) / $(Alg.N) >>>")
            print("\n")
            show(l2rinfo)
            flush(stdout)

            merge!(localto,mto)
            merge!(info,l2rinfo)

            r2linfo = Algebrasweepinfo(R2L())
            mto = mul!(EnvAB,α,Alg,r2linfo)

            show(mto;title = "<<< mul! - $(info.n) / $(Alg.N) <<<")
            print("\n")
            show(r2linfo)
            flush(stdout)

            merge!(localto,mto)
            merge!(info,r2linfo)

            _merge_io!(localto)
            merge!(to,localto)

            info.err < Alg.tol && break
        end
        return xp!(C′, C), to, info
    finally
        Alg.isdisk && (cleanup!(EnvAB); cleanup!(C′))
    end

end

function mul!(EnvAB::Environment{3}, α::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R}; kwargs...)
    localto = TimerOutput()
    L = length(EnvAB.layer[1])
    for site in 1:L-1
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(EnvAB.layer[3][site:site+1]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "projection" projH = proj2(EnvAB, site, site+1)
        @timeit localto "action" t = actionb(projH, composite(EnvAB.layer[1][site:site+1]...))
        @timeit localto "SVD" tl, tc, tr, localinfo.truncerr, localinfo.bond = tsvd(axpy!(α,t,nothing); direction=:center,trunc = Alg.trunc)
        @timeit localto "splice" tr = splice(tc,tr) 
        @timeit localto "push right" begin
            EnvAB.layer[3][site:site+1] = adjoint.([tl, tr])
            map(v -> canonicalize!(EnvAB.layer[v],site + 1),1:3)
            pushright!(EnvAB)
        end
        x = composite(EnvAB.layer[3][site:site+1]...)
        localinfo.err = norm(x-x₀)^2/x2
        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
    end

    return localto
end

function mul!(EnvAB::Environment{3}, α::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L}; kwargs...)
    localto = TimerOutput()
    L = length(EnvAB.layer[1])
    for site in L:-1:2
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(EnvAB.layer[3][site-1:site]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "projection" projH = proj2(EnvAB, site-1, site)
        @timeit localto "action" t = actionb(projH, composite(EnvAB.layer[1][site-1:site]...))
        @timeit localto "SVD" tl, tc, tr, localinfo.truncerr, localinfo.bond = tsvd(axpy!(α, t, nothing); direction=:center,trunc = Alg.trunc)
        @timeit localto "splice" tl = splice(tl,tc) 
        @timeit localto "push left" begin
            EnvAB.layer[3][site-1:site] = adjoint.([tl, tr])
            map(v -> canonicalize!(EnvAB.layer[v],site - 1),1:3)
            pushleft!(EnvAB)
        end
        x = composite(EnvAB.layer[3][site-1:site]...)
        localinfo.err = norm(x-x₀)^2/x2
        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
    end

    return localto
end

function mul!(EnvAB::Environment{3}, α::Number, Alg::Algebraalgo{SingleSite,alg}, sweepinfo::Algebrasweepinfo{L2R}; kwargs...) where alg
    localto = TimerOutput()
    L = length(EnvAB.layer[1])
    for site in 1:L-1
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite((EnvAB.layer[3][site:site+1])...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(L2R())
            @timeit localto "CBE_AB" cbetoAB = CBE!(EnvAB, Alg.alg, cbeinfo)
            merge!(localinfo,cbeinfo)
            merge!(localto,cbetoAB,tree_point = ["CBE_AB"])
        end
        @timeit localto "projection" projH = proj1(EnvAB,site)
        @timeit localto "action" t = actionb(projH,EnvAB.layer[1][site])
        @timeit localto "orthogonalize" begin
            # tl,tr = leftorth(axpy!(α, t, nothing))
            tl,tr,localinfo.truncerr,localinfo.bond = tsvd(axpy!(α, t, nothing); direction=:right,trunc = Alg.trunc)
            tr = splice(tr,EnvAB.layer[3][site+1]')
            EnvAB.layer[3][site:site+1] = adjoint.([tl, tr])
        end
        @timeit localto "push right" begin
            map(v -> canonicalize!(EnvAB.layer[v],site + 1),1:3)
            pushright!(EnvAB)
        end

        x = composite((EnvAB.layer[3][site:site+1])...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
    end

    return localto
end

function mul!(EnvAB::Environment{3}, α::Number, Alg::Algebraalgo{SingleSite,alg}, sweepinfo::Algebrasweepinfo{R2L}; kwargs...) where alg
    localto = TimerOutput()
    L = length(EnvAB.layer[1])
    for site in L:-1:2
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite((EnvAB.layer[3][site-1:site])...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(R2L())
            @timeit localto "CBE_AB" cbetoAB = CBE!(EnvAB, Alg.alg, cbeinfo)
            merge!(localinfo,cbeinfo)
            merge!(localto,cbetoAB,tree_point = ["CBE_AB"])
        end
        @timeit localto "projection" projH = proj1(EnvAB,site)
        @timeit localto "action" t = actionb(projH,EnvAB.layer[1][site])
        @timeit localto "orthogonalize" begin
            # tl,tr = rightorth(axpy!(α, t, nothing))
            tl,tr,localinfo.truncerr,localinfo.bond = tsvd(axpy!(α, t, nothing); direction=:left,trunc = Alg.trunc)
            tl = splice(EnvAB.layer[3][site-1]',tl)
            EnvAB.layer[3][site-1:site] = adjoint.([tl, tr])
        end
        @timeit localto "push left" begin
            map(v -> canonicalize!(EnvAB.layer[v],site - 1),1:3)
            pushleft!(EnvAB)
        end
        x = composite((EnvAB.layer[3][site-1:site])...)
        localinfo.err = norm(x-x₀)^2/x2
        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
    end

    return localto
end


function mul!(C::Union{DenseMPO,DenseMPS}, A::Union{DenseMPO,DenseMPS}, B::Union{DenseMPO,SparseMPO}, α::Number, trunc::TruncationScheme;kwargs...)
    D = _getdim(trunc)
    ϵ = _getcutoff(trunc)

    Nmul = get(kwargs,:Nmul,3)
    verbose = get(kwargs,:verbose,false)
    alg = get(kwargs,:alg,Algebraalgo(SingleSite(),CBEalgo(dynamicSVD(ceil(Int64, D * 1.25)),NoStruc(),0,D),trunc,Nmul,ϵ,verbose))
    
    return mul!(C,A,B,α,alg)
end

