function CBE!(env::Environment, alg::CBEalgo{dynamicSVD}, info::CBEinfo{Dir};kwargs...) where Dir
    to = TimerOutput()

    Dl,Dr = _cbe_maxdim(env,alg,info)
    
    if !(Dl ≤ alg.scheme.Df || Dr ≤ alg.scheme.Df)
        @timeit to "rand SVD" localto = CBE!(env,CBEalgo(alg,randSVD(alg.scheme.Df)),info)
        merge!(to,localto,tree_point = ["rand SVD"])
        return to
    end

    Dc = _cbe_currentdim(env,alg,info)
    if !(Dl ≤ Dc || Dr ≤ Dc)
        @timeit to "full SVD" localto = CBE!(env,CBEalgo(alg,fullSVD()),info)
        merge!(to,localto,tree_point = ["full SVD"])
        return to
    end
    return to
end

CBE!(env::CBEenvironment, alg::CBEalgo{randSVD}, info::CBEinfo;kwargs...) = randSVD!(env,alg,info)
CBE!(env::CBEenvironment, alg::CBEalgo{fullSVD}, info::CBEinfo;kwargs...) = fullSVD!(env,alg,info)

