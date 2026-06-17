function CBE!(env::Environment, alg::CBEalgo{dynamicSVD}, info::CBEinfo;kwargs...)
    site = env.center[1]
    to = TimerOutput()
    if min(site,length(env.layer[1]) - site) ≤ alg.scheme.N
        @timeit to "full SVD" localto = CBE!(env,CBEalgo(alg,fullSVD()),info)
        merge!(to,localto,tree_point = ["full SVD"])
    else
        @timeit to "rand SVD" localto = CBE!(env,CBEalgo(alg,randSVD(alg.scheme.λ)),info)
        merge!(to,localto,tree_point = ["rand SVD"])
    end
    return to
end

CBE!(env::CBEenvironment, alg::CBEalgo{randSVD}, info::CBEinfo;kwargs...) = randSVD!(env,alg,info)
CBE!(env::CBEenvironment, alg::CBEalgo{fullSVD}, info::CBEinfo;kwargs...) = fullSVD!(env,alg,info)



