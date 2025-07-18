
function DMRG1!(ψ::DenseMPS,H::SparseMPO;kwargs...)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ'])
        initialize!(Env)
    end
    flush(stdout)
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,DMRGDefaultLanczos)
    N = get(kwargs,:N,5)
    tol = get(kwargs,:tol,Inf)
    λ = get(kwargs,:λ,1.2)
    Nfull = get(kwargs,:Nfull,4)
    subalgo = get(kwargs,:subalgo,CBEalgo(dynamicSVD(λ,Nfull),DSA(),1,_getdim(trunc)))
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = DMRGalgo(SingleSite(),subalgo,trunc,N,tol,solver,GCsweep,GCsite)
    lsE,lsinfo = DMRG!(Env,alg)
    return lsE,lsinfo
end

function DMRG2!(ψ::DenseMPS,H::SparseMPO;kwargs...)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ'])
        initialize!(Env)
    end
    flush(stdout)
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,DMRGDefaultLanczos)
    N = get(kwargs,:N,5)
    tol = get(kwargs,:tol,Inf)
    subalgo = get(kwargs,:subalgo,NoAlgorithm())
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = DMRGalgo(DoubleSite(),subalgo,trunc,N,tol,solver,GCsweep,GCsite)
    lsE,lsinfo = DMRG!(Env,alg)
    return lsE,lsinfo
end


function DMRG!(Env::Environment{3,L}, Alg::DMRGalgo;kwargs...) where L
    info =  DMRGinfo()
    lsinfo = []
    lsE = []
    for _ in 1:Alg.N

        l2rinfo = DMRGsweepinfo(L2R())
        to = DMRG!(Env,Alg,l2rinfo)
        show(to;title=">>> DMRG >>>")
        print("\n")
        show(l2rinfo)
        merge!(info,l2rinfo)
        flush(stdout)

        r2linfo = DMRGsweepinfo(R2L())
        to = DMRG!(Env,Alg,r2linfo)
        show(to;title="<<< DMRG <<<")
        print("\n")
        show(r2linfo)
        merge!(info,r2linfo)
        flush(stdout)

        push!(lsinfo,deepcopy(info))
        push!(lsE,info.E)

        info.err > Alg.tol && return lsE,lsinfo
    end
    return lsE,lsinfo
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{SingleSite,alg},info::DMRGsweepinfo{L2R}) where {L,alg}
    localto = TimerOutput()
    lsE = []
    for site in 1:L-1
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(L2R())
            @timeit localto "CBE" cbeto = CBE!(Env, Alg.alg, cbeinfo)
            merge!(localinfo,cbeinfo)
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj1(Env,site;E₀ = E₀)
            Eg, Egv, localinfo.solver = groundEig(projH;x₀ = Env.layer[1].ts[site])
            localinfo.E = E₀ + Eg |> real
        end
        merge!(localto,get_timer("action");tree_point = ["Krylov"])
        @timeit localto "orthogonalize" begin
            @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc)
            localinfo.bond = BondInfo(tc)
            @timeit localto "contract" tr = contract(contract(tc,tr),Env.layer[1].ts[site+1])
        end
        @timeit localto "pushright" pushright!(Env,tl, tr)
        push!(lsE,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end

    info.σE = std(filter(!isnan,lsE))

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{SingleSite,alg},info::DMRGsweepinfo{R2L}) where {L,alg}
    localto = TimerOutput()
    lsE = []
    for site in L:-1:2
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(R2L())
            @timeit localto "CBE" cbeto = CBE!(Env, Alg.alg, cbeinfo)
            merge!(localinfo,cbeinfo)
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj1(Env,site;E₀ = E₀)
            Eg, Egv, localinfo.solver = groundEig(projH;x₀ = Env.layer[1].ts[site])
            localinfo.E = E₀ + Eg |> real
        end
        merge!(localto,get_timer("action");tree_point = ["Krylov"])
        @timeit localto "orthogonalize" begin
            @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc,index_tuple = ((1,),(2,3)))
            tr = MPSTensor(permute(tr.A,(1,2),(3,)))
            localinfo.bond = BondInfo(tc)
            @timeit localto "contract" tl = contract(Env.layer[1].ts[site-1],contract(tl,tc))
        end
        @timeit localto "pushleft" pushleft!(Env,tl, tr)
        push!(lsE,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end

    info.σE = std(filter(!isnan,lsE))

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{DoubleSite},info::DMRGsweepinfo{L2R}) where L
    localto = TimerOutput()
    lsE = []
    for site in 1:L-1
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj2(Env,site,site+1;E₀ = E₀)
            @timeit localto "composite" x₀ = composite(Env.layer[1].ts[site:site+1]...)
            Eg,Egv,localinfo.solver = groundEig(projH;x₀ = x₀)
            localinfo.E = E₀ + Eg |> real
        end
        merge!(localto,get_timer("action");tree_point = ["Krylov"])
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tr = contract(tc,tr) 
        @timeit localto "pushright" pushright!(Env,tl, tr)
        push!(lsE,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end
    info.σE = std(filter(!isnan,lsE))

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{DoubleSite},info::DMRGsweepinfo{R2L}) where L
    localto = TimerOutput()
    lsE = []
    for site in L:-1:2
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj2(Env,site-1,site;E₀ = E₀)
            @timeit localto "composite" x₀ = composite(Env.layer[1].ts[site-1:site]...)
            Eg,Egv,localinfo.solver = groundEig(projH;x₀ = x₀)
            localinfo.E = E₀ + Eg |> real
        end 
        merge!(localto,get_timer("action");tree_point = ["Krylov"]) 
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tl = contract(tl,tc) 
        @timeit localto "pushleft" pushleft!(Env,tl, tr)
        push!(lsE,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end
    info.σE = std(filter(!isnan,lsE))

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

