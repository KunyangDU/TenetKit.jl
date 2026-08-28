
function DMRG1!(ψ::DenseMPS,H::Union{DenseMPO,SparseMPO};kwargs...)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    ψ′ = RefMPS(ψ)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ′];isdisk=isdisk)
        initialize!(Env)
    end
    flush(stdout)
    trunc = get(kwargs,:trunc,notrunc())
    solver = get(kwargs,:solver,DMRGDefaultLanczos)
    N = get(kwargs,:N,20)
    Etol = get(kwargs,:Etol,1e-7)
    Stol = get(kwargs,:Stol,1e-6)
    subalgo = get(kwargs,:subalgo,CBEalgo(dynamicSVD(ceil(Int64,_getdim(trunc) * 1.25)),issparse(H) ? DSA() : DDA(),1,_getdim(trunc)))
    GCsweep = get(kwargs, :GCsweep, true)
    GCsite = get(kwargs, :GCsite, false)
    verbose = get(kwargs, :verbose, false)
    alg = DMRGalgo(SingleSite(),subalgo,trunc,N,Etol,Stol,solver,GCsweep,GCsite,verbose,isdisk)
    try
        lsE,lsinfo = DMRG!(Env,alg)
        return lsE,lsinfo
    finally
        cleanup!(Env)
    end
end

function DMRG2!(ψ::DenseMPS,H::Union{DenseMPO,SparseMPO};kwargs...)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    ψ′ = RefMPS(ψ)
    @time "initialize environment" begin
        Env = Environment([ψ,H,ψ′];isdisk=isdisk)
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
    verbose = get(kwargs, :verbose, false)
    alg = DMRGalgo(DoubleSite(),subalgo,trunc,N,Etol,Stol,solver,GCsweep,GCsite,verbose,isdisk)
    try
        lsE,lsinfo = DMRG!(Env,alg)
        return lsE,lsinfo
    finally
        cleanup!(Env)
    end
end
