
function TDVP1!(Env::Environment{3}, lst::AbstractVector;kwargs...)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])
    info = TDVPinfo()
    lsinfo = []
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,1e-4)
    λ = get(kwargs,:λ,1.2)
    subalgo = get(kwargs,:subalgo,CBEalgo(randSVD(),λ,_getdim(trunc),_getcutoff(trunc)))
    alg = TDVPalgo(SingleSite(),subalgo,trunc,0,tol,solver)
    
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
    alg = TDVPalgo(DoubleSite(),subalgo,trunc,0,tol,solver)
    
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

    l2rinfo = TDVPsweepinfo(L2R(),info.err)
    to = TDVP!(Env,Alg,l2rinfo)
    show(to;title=">>> TDVP >>>")
    print("\n")
    show(l2rinfo)
    merge!(info,l2rinfo)
    flush(stdout)

    if isreal(Alg.τ)
        @assert (d = normalize!(Env.layer[1])) ≈ normalize!(Env.layer[3])
        info.lnZ += 2 * log(d)
    end

    r2linfo = TDVPsweepinfo(R2L(),info.err)
    to = TDVP!(Env,Alg,r2linfo)
    show(to;title="<<< TDVP <<<")
    print("\n")
    show(r2linfo)
    merge!(info,r2linfo)
    flush(stdout)

    if isreal(Alg.τ)
        @assert (d = normalize!(Env.layer[1])) ≈ normalize!(Env.layer[3])
        info.lnZ += 2 * log(d)
        info.E = real(scalar(Env))
    end

    GC.gc()
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{DoubleSite},info::TDVPsweepinfo{L2R}) where L
    localto = TimerOutput()
    for site in 1:L-1
        localinfo = TDVPsiteinfo()
        @timeit localto "evolve" tmp,localinfo.solver = evolve!(composite(Env.layer[1].ts[site:site+1]...), proj2(Env,site,site+1), Alg.τ, Alg.solver)
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(tmp; direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        tr = contract(tc,tr)
        to,solver = pushright!(Env, tl, tr, Alg.τ, Alg.solver)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end    
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[L], proj1(Env,L), Alg.τ, Alg.solver)
    Env.layer[3].ts[L] = Env.layer[1].ts[L]'
    merge!(info.solver, solver)
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{DoubleSite},info::TDVPsweepinfo{R2L}) where L
    localto = TimerOutput()
    for site in L:-1:2
        localinfo = TDVPsiteinfo()
        @timeit localto "evolve" tmp, localinfo.solver = evolve!(composite(Env.layer[1].ts[site-1:site]...), proj2(Env,site-1,site), Alg.τ, Alg.solver)
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(tmp; direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        tl = contract(tl,tc)
        to,solver = pushleft!(Env, tl, tr, Alg.τ, Alg.solver)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[1], proj1(Env,1), Alg.τ)
    Env.layer[3].ts[1] = Env.layer[1].ts[1]'
    merge!(info.solver, solver)
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{SingleSite,alg},info::TDVPsweepinfo{L2R}) where {L,alg}
    localto = TimerOutput()
    for site in 1:L-1
        localinfo = TDVPsiteinfo()
        if alg <: CBEalgo 
            @timeit localto "CBE" begin
                cbeinfo = CBEinfo(L2R())
                B = deepcopy(Env.layer[1].ts[site+1])
                cbeto = CBE!(Env,Alg.alg,cbeinfo)
                splice!(Env.layer[1],B,site+1)
                Env.layer[3].ts[site] = Env.layer[1].ts[site]'
                merge!(localinfo,cbeinfo)
            end
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "evolve" tmp,localinfo.solver = evolve!(Env.layer[1].ts[site], proj1(Env,site), Alg.τ, Alg.solver)
        @timeit localto "orthogonalize" begin
            tl,tr = leftorth(tmp)
            localinfo.bond = BondInfo(tr)
        end 
        to,solver = pushright!(Env,tl,tr,Alg.τ,Alg.solver)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[L], proj1(Env,L), Alg.τ, Alg.solver)
    Env.layer[3].ts[L] = Env.layer[1].ts[L]'
    merge!(info.solver, solver)
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{SingleSite,alg},info::TDVPsweepinfo{R2L}) where {L,alg}
    localto = TimerOutput()
    for site in L:-1:2
        localinfo = TDVPsiteinfo()
        if alg <: CBEalgo 
            @timeit localto "CBE" begin
                cbeinfo = CBEinfo(R2L())
                A = deepcopy(Env.layer[1].ts[site-1])
                cbeto = CBE!(Env,Alg.alg,cbeinfo)
                splice!(Env.layer[1],A,site-1)
                Env.layer[3].ts[site] = Env.layer[1].ts[site]'
                merge!(localinfo,cbeinfo)
            end
            merge!(localto,cbeto,tree_point = ["CBE"])
        end
        @timeit localto "evolve" tmp, localinfo.solver = evolve!(Env.layer[1].ts[site], proj1(Env,site), Alg.τ, Alg.solver)
        @timeit localto "orthogonalize" begin
            tl,tr = rightorth(tmp)
            localinfo.bond = BondInfo(tl)
        end
        to,solver = pushleft!(Env,tl,tr,Alg.τ)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[1], proj1(Env,1), Alg.τ, Alg.solver)
    Env.layer[3].ts[1] = Env.layer[1].ts[1]'
    merge!(info.solver, solver)
    return localto
end

function TDVP1!(ψ::DenseMPS,H::SparseMPO,t::Number,Nt::Number;kwargs...)
    @time "initialize environment" begin 
        Env = Environment([ψ,H,ψ'])
        initialize!(Env)
    end
    flush(stdout)
    lst = range(0,t,Nt)
    lsobj,lsinfo = TDVP1!(Env,1im * lst;kwargs...)
    return lst,lsobj,lsinfo
end

function TDVP2!(ψ::DenseMPS,H::SparseMPO,t::Number,Nt::Number;kwargs...)
    @time "initialize environment" begin 
        Env = Environment([ψ,H,ψ'])
        initialize!(Env)
    end
    flush(stdout)
    lst = range(0,t,Nt)
    lsobj,lsinfo = TDVP2!(Env,1im * lst;kwargs...)
    return lst,lsobj,lsinfo
end


function tanTRG1!(ρ::DenseMPO,H::SparseMPO,lsβ::Vector;kwargs...)
    @time "initialize environment" begin 
        Env = Environment([ρ,H,ρ'])
        initialize!(Env)
    end
    flush(stdout)
    return tanTRG1!(Env, lsβ;kwargs...)
end
function tanTRG2!(ρ::DenseMPO,H::SparseMPO,lsβ::Vector;kwargs...)
    @time "initialize environment" begin 
        Env = Environment([ρ,H,ρ'])
        initialize!(Env)
    end
    flush(stdout)
    return tanTRG2!(Env, lsβ;kwargs...)
end

function tanTRG2!(Env::Environment{3}, lsβ::AbstractVector;kwargs...)

    lnZ = get(kwargs,:lnZ,0.0)
    info = TDVPinfo(lnZ)
    lsinfo = []

    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,Inf)
    subalgo = get(kwargs,:subalgo,NoAlgorithm())
    alg = TDVPalgo(DoubleSite(),subalgo,trunc,0,tol,solver)

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
    subalgo = get(kwargs,:subalgo,CBEalgo(randSVD(),λ,_getdim(trunc),_getcutoff(trunc)))
    alg = TDVPalgo(SingleSite(),subalgo,trunc,0,tol,solver)

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

