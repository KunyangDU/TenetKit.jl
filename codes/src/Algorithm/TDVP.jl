
function TDVP1!(Env::Environment{3}, lst::AbstractVector;kwargs...)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])
    info = TDVPinfo()
    lsinfo = []
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,1e-4)
    λ = get(kwargs,:λ,1.2)
    Nfull = get(kwargs,:Nfull,-1)
    subalgo = get(kwargs,:subalgo,CBEalgo(dynamicSVD(λ,Nfull),DSA(),1,_getdim(trunc)))
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = TDVPalgo(SingleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite)
    
    for i in 2:length(lst)
        τ = (lst[i]-lst[i-1])/2
        println("t = $(abs(lst[i]))")
        flush(stdout)
        alg.τ = τ
        
        TDVP!(Env, alg, info)

        info.err > alg.tol && break
        push!(lsobj,deepcopy(Env.layer[1]))
        push!(lsinfo,deepcopy(info))
    end

    return lsobj,lsinfo
end

function TDVP2!(Env::Environment{3}, lst::AbstractVector;kwargs...)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])
    info = TDVPinfo()
    lsinfo = []
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,1e-4)
    subalgo = get(kwargs,:subalgo,NoAlgorithm())
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = TDVPalgo(DoubleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite)
    
    for i in 2:length(lst)
        τ = (lst[i]-lst[i-1])/2
        println("t = $(abs(lst[i]))")
        flush(stdout)
        alg.τ = τ
        
        TDVP!(Env, alg, info)

        info.err > alg.tol && break
        push!(lsobj,deepcopy(Env.layer[1]))
        push!(lsinfo,deepcopy(info))
    end

    return lsobj,lsinfo
end

function TDVP!(Env::Environment{3,L}, Alg::TDVPalgo, info::TDVPinfo;kwargs...) where L

    iszero(info.E) && (info.E = _scalar(Env) |> real)
    __init_io__()

    l2rinfo = TDVPsweepinfo(L2R(),info.err)
    l2rinfo.E = info.E
    to = TDVP!(Env,Alg,l2rinfo)
    if isreal(Alg.τ)
        Env.layer[3] isa RefMPO ? (d = normalize!(Env.layer[1])) : (@assert (d = normalize!(Env.layer[1])) ≈ normalize!(Env.layer[3]))
        info.lnZ += 2 * log(d)
    end
    _merge_io!(to)
    show(to;title=">>> TDVP >>>")
    print("\n")
    show(l2rinfo)
    merge!(info,l2rinfo)
    flush(stdout)

    r2linfo = TDVPsweepinfo(R2L(),info.err)
    r2linfo.E = info.E
    to = TDVP!(Env,Alg,r2linfo)
    if isreal(Alg.τ)
        Env.layer[3] isa RefMPO ? (d = normalize!(Env.layer[1])) : (@assert (d = normalize!(Env.layer[1])) ≈ normalize!(Env.layer[3]))
        info.lnZ += 2 * log(d)
    end
    _merge_io!(to)
    show(to;title="<<< TDVP <<<")
    print("\n")
    show(r2linfo)
    merge!(info,r2linfo)
    info.E = _scalar(Env) |> real 
    println("ΔE/τ = (Er - El)/(τ/2) = $((info.E - r2linfo.E)/abs(Alg.τ/2))")
    flush(stdout)
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{DoubleSite},info::TDVPsweepinfo{L2R}) where L
    localto = TimerOutput()
    for site in 1:L-1
        localinfo = TDVPsiteinfo()
        @timeit localto "evolve" tmp,localinfo.solver = evolve!(composite(Env.layer[1][site:site+1]...), proj2(Env,site,site+1;E₀ = info.E), Alg.τ, Alg.solver)
        merge!(localto,get_timer("action");tree_point = ["evolve"])
        rmul!(tmp,exp(-Alg.τ * info.E))
        nmt = normalize!(tmp)
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(tmp; direction=:center,trunc = Alg.trunc)
        merge!(localinfo,BondInfo(tc))
        tr = rmul!(contract(tc,tr),nmt)
        to,solver = pushright!(Env, tl, tr, Alg, info)
        merge!(localto,to)
        merge!(localto,get_timer("action");tree_point = ["back evolve"])
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)

        Alg.GCsite && @timeit localto "GC" GC.gc()
    end    
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1][L], proj1(Env,L;E₀ = info.E), Alg.τ, Alg.solver)
    rmul!(Env.layer[1][L],exp(-Alg.τ * info.E))
    merge!(localto,get_timer("action");tree_point = ["evolve"])
    Env.layer[3][L] = Env.layer[1][L]'
    merge!(info.solver, solver)

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{DoubleSite},info::TDVPsweepinfo{R2L}) where L
    localto = TimerOutput()
    for site in L:-1:2
        localinfo = TDVPsiteinfo()
        @timeit localto "evolve" tmp, localinfo.solver = evolve!(composite(Env.layer[1][site-1:site]...), proj2(Env,site-1,site;E₀ = info.E), Alg.τ, Alg.solver)
        merge!(localto,get_timer("action");tree_point = ["evolve"])
        rmul!(tmp,exp(-Alg.τ * info.E))
        nmt = normalize!(tmp)
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(tmp; direction=:center,trunc = Alg.trunc)
        merge!(localinfo,BondInfo(tc))
        tl = rmul!(contract(tl,tc),nmt)
        to,solver = pushleft!(Env, tl, tr, Alg, info)
        merge!(localto,to)
        merge!(localto,get_timer("action");tree_point = ["back evolve"])
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)

        Alg.GCsite && @timeit localto "GC" GC.gc()
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1][1], proj1(Env,1;E₀ = info.E), Alg.τ)
    rmul!(Env.layer[1][1],exp(-Alg.τ * info.E))
    merge!(localto,get_timer("action");tree_point = ["evolve"])
    Env.layer[3][1] = Env.layer[1][1]'
    merge!(info.solver, solver)

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{SingleSite,alg},info::TDVPsweepinfo{L2R}) where {L,alg}
    localto = TimerOutput()
    for site in 1:L-1
        localinfo = TDVPsiteinfo()
        if alg <: CBEalgo 
            @timeit localto "CBE" begin
                cbeinfo = CBEinfo(L2R())
                cbeto = CBE!(Env,Alg.alg,cbeinfo)
                merge!(localinfo,cbeinfo)
            end
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "evolve" tmp,localinfo.solver = evolve!(Env.layer[1][site], proj1(Env,site;E₀ = info.E), Alg.τ, Alg.solver)
        merge!(localto,get_timer("action");tree_point = ["evolve"])
        rmul!(tmp,exp(-Alg.τ * info.E))
        nmt = normalize!(tmp)
        @timeit localto "orthogonalize" begin
            tl,tc,tr,localinfo.err,svdto = _tdvp_tsvd(tmp,Alg.trunc,L2R())
            merge!(localto,svdto;tree_point = ["orthogonalize"])
            merge!(localinfo,BondInfo(tc))
        end 
        to,solver = pushright!(Env,tl,rmul!(tr,nmt),Alg,info)
        merge!(localto,to)
        merge!(localto,get_timer("action");tree_point = ["back evolve"])
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)

        Alg.GCsite && @timeit localto "GC" GC.gc()
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1][L], proj1(Env,L;E₀ = info.E), Alg.τ, Alg.solver)
    rmul!(Env.layer[1][L],exp(-Alg.τ * info.E))
    merge!(localto,get_timer("action");tree_point = ["evolve"])
    Env.layer[3][L] = Env.layer[1][L]'
    merge!(info.solver, solver)

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{SingleSite,alg},info::TDVPsweepinfo{R2L}) where {L,alg}
    localto = TimerOutput()
    for site in L:-1:2
        localinfo = TDVPsiteinfo()
        if alg <: CBEalgo 
            @timeit localto "CBE" begin
                cbeinfo = CBEinfo(R2L())
                cbeto = CBE!(Env,Alg.alg,cbeinfo)
                merge!(localinfo,cbeinfo)
            end
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "evolve" tmp, localinfo.solver = evolve!(Env.layer[1][site], proj1(Env,site;E₀ = info.E), Alg.τ, Alg.solver)
        merge!(localto,get_timer("action");tree_point = ["evolve"])
        rmul!(tmp,exp(-Alg.τ * info.E))
        nmt = normalize!(tmp)
        @timeit localto "orthogonalize" begin
            tl,tc,tr,localinfo.err,svdto = _tdvp_tsvd(tmp,Alg.trunc,R2L())
            merge!(localto,svdto;tree_point = ["orthogonalize"])
            merge!(localinfo,BondInfo(tc))
        end
        to,solver = pushleft!(Env,rmul!(tl,nmt),tr,Alg,info)
        merge!(localto,to)
        merge!(localto,get_timer("action");tree_point = ["back evolve"])
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)

        Alg.GCsite && @timeit localto "GC" GC.gc()
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1][1], proj1(Env,1;E₀ = info.E), Alg.τ, Alg.solver)
    rmul!(Env.layer[1][1],exp(-Alg.τ * info.E))
    Env.layer[3][1] = Env.layer[1][1]'
    merge!(info.solver, solver)
    merge!(localto,get_timer("action");tree_point = ["evolve"])

    Alg.GCsweep && @timeit localto "GC" GC.gc()
    return localto
end

function TDVP1!(ψ::DenseMPS,H::SparseMPO,t::Number,Nt::Number;kwargs...)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    ψ′ = RefMPS(ψ)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ′];isdisk=isdisk)
        initialize!(Env)
    end
    flush(stdout)
    lst = range(0,t,Nt)
    try
        lsobj,lsinfo = TDVP1!(Env,1im * lst;kwargs...)
        return lst,lsobj,lsinfo
    finally
        cleanup!(Env)
    end
end

function TDVP2!(ψ::DenseMPS,H::SparseMPO,t::Number,Nt::Number;kwargs...)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    ψ′ = RefMPS(ψ)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ′];isdisk=isdisk)
        initialize!(Env)
    end
    flush(stdout)
    lst = range(0,t,Nt)
    try
        lsobj,lsinfo = TDVP2!(Env,1im * lst;kwargs...)
        return lst,lsobj,lsinfo
    finally
        cleanup!(Env)
    end
end


function tanTRG1!(ρ::DenseMPO,H::SparseMPO,lsβ::Vector;kwargs...)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    ρ′ = RefMPO(ρ, adjoint)
    @time "initialize environment" begin
        Env = Environment([ρ,H,ρ′];isdisk=isdisk)
        initialize!(Env)
    end
    flush(stdout)
    try
        return tanTRG1!(Env, lsβ;kwargs...)
    finally
        cleanup!(Env)
    end
end
function tanTRG2!(ρ::DenseMPO,H::SparseMPO,lsβ::Vector;kwargs...)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    ρ′ = RefMPO(ρ, adjoint)
    @time "initialize environment" begin
        Env = Environment([ρ,H,ρ′];isdisk=isdisk)
        initialize!(Env)
    end
    flush(stdout)
    try
        return tanTRG2!(Env, lsβ;kwargs...)
    finally
        cleanup!(Env)
    end
end

function tanTRG2!(Env::Environment{3}, lsβ::AbstractVector;kwargs...)

    lnZ = get(kwargs,:lnZ,0.0)
    info = TDVPinfo(lnZ)
    lsinfo = []

    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,Inf)
    subalgo = get(kwargs,:subalgo,NoAlgorithm())
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = TDVPalgo(DoubleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite)

    lsobj = []
    lsF = Float64[]
    lsE = Float64[]

    for i in 2:length(lsβ)
        τ = (lsβ[i]-lsβ[i-1])/2
        println("t = $(lsβ[i])")
        flush(stdout)
        alg.τ = τ
        
        TDVP!(Env, alg, info)

        info.err > alg.tol && break
        push!(lsobj,deepcopy(Env.layer[1]))
        push!(lsinfo,deepcopy(info))
        push!(lsF, - info.lnZ / lsβ[i] / 2)
        push!(lsE, info.E)
    end

    return lsobj,lsinfo,lsF,lsE
end

function tanTRG1!(Env::Environment{3}, lsβ::AbstractVector;kwargs...)
    
    lnZ = get(kwargs,:lnZ,0.0)
    info = TDVPinfo(lnZ)
    lsinfo = []

    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,Inf)
    λ = get(kwargs,:λ,1.2)
    subalgo = get(kwargs,:subalgo,CBEalgo(randSVD(λ),DSA(),1,_getdim(trunc)))
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    alg = TDVPalgo(SingleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite)

    lsobj = []
    lsF = Float64[]
    lsE = Float64[]
    
    for i in 2:length(lsβ)
        τ = (lsβ[i]-lsβ[i-1])/2
        println("t = $( lsβ[i])")
        flush(stdout)
        alg.τ = τ
        
        TDVP!(Env, alg, info)

        info.err > alg.tol && break
        push!(lsobj,deepcopy(Env.layer[1]))
        push!(lsinfo,deepcopy(info))
        push!(lsF, - info.lnZ / lsβ[i] / 2)
        push!(lsE, info.E)
    end

    return lsobj,lsinfo,lsF,lsE
end

