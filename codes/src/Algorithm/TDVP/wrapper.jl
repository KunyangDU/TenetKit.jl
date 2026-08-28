

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

function TDVP1!(Env::Environment{3}, lst::AbstractVector;kwargs...)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])
    info = TDVPinfo()
    lsinfo = []
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,1e-4)
    subalgo = get(kwargs,:subalgo,CBEalgo(dynamicSVD(ceil(Int64,_getdim(trunc) * 1.25)),DSA(),1,_getdim(trunc)))
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    verbose = get(kwargs, :verbose, false)
    alg = TDVPalgo(SingleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite,verbose)

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
    verbose = get(kwargs, :verbose, false)
    alg = TDVPalgo(DoubleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite,verbose)
    
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


function tanTRG1!(Env::Environment{3}, lsβ::AbstractVector;kwargs...)
    
    lnZ = get(kwargs,:lnZ,0.0)
    info = TDVPinfo(lnZ)
    lsinfo = []

    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,TDVPDefaultLanczos)
    tol = get(kwargs,:tol,Inf)
    subalgo = get(kwargs,:subalgo,CBEalgo(randSVD(ceil(Int64,1.25 * _getdim(trunc))),DSA(),1,_getdim(trunc)))
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    verbose = get(kwargs, :verbose, false)
    alg = TDVPalgo(SingleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite,verbose)

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
    verbose = get(kwargs, :verbose, false)
    alg = TDVPalgo(DoubleSite(),subalgo,trunc,0,tol,solver,GCsweep,GCsite,verbose)

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
