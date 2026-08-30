
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




