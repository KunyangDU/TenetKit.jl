
function DMRG1!(ψ::DenseMPS,H::Union{DenseMPO,SparseMPO};kwargs...)
    isdisk = get(kwargs,:isdisk,false)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ'];disk=isdisk)
        initialize!(Env)
    end
    flush(stdout)
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,DMRGDefaultLanczos)
    N = get(kwargs,:N,20)
    Etol = get(kwargs,:Etol,1e-7)
    Stol = get(kwargs,:Stol,1e-6)
    λ = get(kwargs,:λ,1.2)
    Nfull = get(kwargs,:Nfull,4)
    subalgo = get(kwargs,:subalgo,CBEalgo(dynamicSVD(λ,Nfull),DSA(),1,_getdim(trunc)))
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = DMRGalgo(SingleSite(),subalgo,trunc,N,Etol,Stol,solver,GCsweep,GCsite,isdisk)
    lsE,lsinfo = DMRG!(Env,alg)
    return lsE,lsinfo
end

function DMRG2!(ψ::DenseMPS,H::Union{DenseMPO,SparseMPO};kwargs...)
    isdisk = get(kwargs,:isdisk,false)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ'];disk=isdisk)
        initialize!(Env)
    end
    flush(stdout)
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,DMRGDefaultLanczos)
    N = get(kwargs,:N,20)
    Etol = get(kwargs,:Etol,1e-7)
    Stol = get(kwargs,:Stol,1e-6)
    subalgo = get(kwargs,:subalgo,NoAlgorithm())
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = DMRGalgo(DoubleSite(),subalgo,trunc,N,Etol,Stol,solver,GCsweep,GCsite,isdisk)
    lsE,lsinfo = DMRG!(Env,alg)
    return lsE,lsinfo
end


function DMRG!(Env::Environment{3,L}, Alg::DMRGalgo;kwargs...) where L
    lsinfo = []
    lsE = []
    Sv = nothing
    __init_io__()
    for i in 1:Alg.N
        info =  DMRGinfo()
        l2rinfo = DMRGsweepinfo(L2R())
        to = DMRG!(Env,Alg,l2rinfo)
        _merge_io!(to)
        show(to;title=">>> DMRG ($(i)/$(Alg.N)) >>>")
        print("\n")
        show(l2rinfo)
        merge!(info,l2rinfo)
        flush(stdout)

        r2linfo = DMRGsweepinfo(R2L())
        to = DMRG!(Env,Alg,r2linfo)
        _merge_io!(to)
        show(to;title="<<< DMRG ($(i)/$(Alg.N)) <<<")
        print("\n")
        show(r2linfo)
        merge!(info,r2linfo)
        ΔE = (r2linfo.E[end] - l2rinfo.E[end]) / (r2linfo.E[end] + l2rinfo.E[end])
        # ΔS = norm((l2rinfo.S .- reverse(r2linfo.S))) / norm((l2rinfo.S .+ reverse(r2linfo.S))/2)
        ΔS = abs(maximum(l2rinfo.S) - maximum(r2linfo.S))/(maximum(l2rinfo.S) + maximum(r2linfo.S))
        println("ΔE = El - Er = $(ΔE), ΔS = |ΔŜ|/|Ŝ| = $(ΔS)")
        flush(stdout)

        push!(lsinfo,deepcopy(info))
        push!(lsE,info.E[end])

        if ΔE < Alg.Etol && ΔS < Alg.Stol
            return lsE,lsinfo
        end
    end
    return lsE,lsinfo
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{SingleSite,alg},info::DMRGsweepinfo{L2R}) where {L,alg}
    localto = TimerOutput()
    # lsE = []
    for site in 1:L-1
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(L2R())
            @timeit localto "CBE" cbeto = CBE!(Env, Alg.alg, cbeinfo)
            normalize!(Env.layer[1][site])
            normalize!(Env.layer[3][site])
            merge!(localinfo,cbeinfo)
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj1(Env,site;E₀ = E₀)
            Eg, Egv, localinfo.solver = groundEig(projH;x₀ = Env.layer[1][site])
            localinfo.E = E₀ + Eg |> real
        end
        merge!(localto,get_timer("action");tree_point = ["Krylov"])
        @timeit localto "orthogonalize" begin
            @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc)
            # localinfo.bond = BondInfo(tc)
            merge!(localinfo, BondInfo(tc))
            @timeit localto "contract" tr = contract(contract(tc,tr),Env.layer[1][site+1])
        end
        @timeit localto "pushright" pushright!(Env,tl, tr)
        # push!(info.E,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end
    # info.σE = std(filter(!isnan,lsE))
    # info.E = lsE[end]
    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{SingleSite,alg},info::DMRGsweepinfo{R2L}) where {L,alg}
    localto = TimerOutput()
    # lsE = []
    for site in L:-1:2
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(R2L())
            @timeit localto "CBE" cbeto = CBE!(Env, Alg.alg, cbeinfo)
            normalize!(Env.layer[1][site])
            normalize!(Env.layer[3][site])
            merge!(localinfo,cbeinfo)
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj1(Env,site;E₀ = E₀)
            Eg, Egv, localinfo.solver = groundEig(projH;x₀ = Env.layer[1][site])
            localinfo.E = E₀ + Eg |> real
        end
        merge!(localto,get_timer("action");tree_point = ["Krylov"])
        @timeit localto "orthogonalize" begin
            @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc,index_tuple = ((1,),(2,3)))
            tr = MPSTensor(permute(tr.A,(1,2),(3,)))
            # localinfo.bond = BondInfo(tc)
            merge!(localinfo, BondInfo(tc))
            @timeit localto "contract" tl = contract(Env.layer[1][site-1],contract(tl,tc))
        end
        @timeit localto "pushleft" pushleft!(Env,tl, tr)
        # push!(lsE,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end
    # info.σE = std(filter(!isnan,lsE))
    # info.E = lsE[end]
    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{DoubleSite},info::DMRGsweepinfo{L2R}) where L
    localto = TimerOutput()
    # lsE = []
    for site in 1:L-1
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj2(Env,site,site+1;E₀ = E₀)
            @timeit localto "composite" x₀ = composite(Env.layer[1][site:site+1]...)
            Eg,Egv,localinfo.solver = groundEig(projH;x₀ = x₀)
            localinfo.E = E₀ + Eg |> real
        end
        merge!(localto,get_timer("action");tree_point = ["Krylov"])
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc)
        # localinfo.bond = BondInfo(tc)
        merge!(localinfo, BondInfo(tc))
        @timeit localto "contract" tr = contract(tc,tr) 
        @timeit localto "pushright" pushright!(Env,tl, tr)
        # push!(lsE,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end
    # info.σE = std(filter(!isnan,lsE))
    # info.E = lsE[end]
    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{DoubleSite},info::DMRGsweepinfo{R2L}) where L
    localto = TimerOutput()
    # lsE = []
    for site in L:-1:2
        localinfo = DMRGsiteinfo()
        E₀ = _scalar(Env) |> real
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj2(Env,site-1,site;E₀ = E₀)
            @timeit localto "composite" x₀ = composite(Env.layer[1][site-1:site]...)
            Eg,Egv,localinfo.solver = groundEig(projH;x₀ = x₀)
            localinfo.E = E₀ + Eg |> real
        end 
        merge!(localto,get_timer("action");tree_point = ["Krylov"]) 
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(Egv; direction=:center,trunc = Alg.trunc)
        # localinfo.bond = BondInfo(tc)
        merge!(localinfo, BondInfo(tc))
        @timeit localto "contract" tl = contract(tl,tc) 
        @timeit localto "pushleft" pushleft!(Env,tl, tr)
        # push!(lsE,localinfo.E)
        Alg.GCsite && @timeit localto "GC" GC.gc()
        merge!(info,localinfo)
    end
    # info.σE = std(filter(!isnan,lsE))
    # info.E = lsE[end]
    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

