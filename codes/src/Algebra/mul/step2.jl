
function mul!(EnvAB::Environment{3,L}, α::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R}; kwargs...) where L
    localto = TimerOutput()

    for site in 1:L-1
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = composite(EnvAB.layer[1][site:site+1]...)
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "projection" projH = proj2(EnvAB, site, site+1)
        @timeit localto "action" t = actionb(projH, EnvAB.layer[3][site:site+1]...)
        @timeit localto "SVD" EnvAB.layer[1][site], EnvAB.layer[1][site+1], localinfo.truncerr, localinfo.bond = tsvd(rmul!(t,α)'; direction=:right,trunc = Alg.trunc)
        x = composite(EnvAB.layer[1][site:site+1]...)
        EnvAB.layer[1].center .+= 1
        @timeit localto "push right" canonicalize!(EnvAB,site + 1)
        localinfo.err = norm(x-x₀)^2/x2
        @show norm(x-x₀)^2, x2
        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
    end

    return localto
end

function mul!(EnvAB::Environment{3,L}, α::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L}; kwargs...) where L
    localto = TimerOutput()

    for site in L:-1:2
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = composite(EnvAB.layer[1][site-1:site]...)
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "projection" projH = proj2(EnvAB, site-1, site)
        @timeit localto "action" t = actionb(projH, EnvAB.layer[3][site-1:site]...)
        @timeit localto "SVD" EnvAB.layer[3][site-1], EnvAB.layer[3][site], localinfo.truncerr, localinfo.bond = tsvd(rmul!(t, α)'; direction=:left,trunc = Alg.trunc)
        EnvAB.layer[1].center .-= 1
        @timeit localto "push left" canonicalize!(EnvAB,site - 1)
        x = composite(EnvAB.layer[1][site-1:site]...)
        localinfo.err = norm(x-x₀)^2/x2
        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
    end

    return localto
end

