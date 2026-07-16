struct NoAlgorithm <: AbstractAlgorithm NoAlgorithm() = new() end
# struct noalgorithm <: NoAlgorithm noalgorithm() = new() end


struct Krylovalgo <: SolverAlgo
    Alg::KrylovKit.KrylovAlgorithm
    Krylovalgo(Alg::KrylovKit.KrylovAlgorithm) = new(Alg)
end

struct SETTNalgo{Sch} <: AbstractAlgorithm where {Sch}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    trunc::TruncationScheme
    N::Int64
    tol::Number
    function SETTNalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme, N::Int64, tol::Number)
        new{typeof(scheme)}(scheme,alg,trunc,N,tol)
    end
end

struct DMRGalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    trunc::TruncationScheme
    N::Int64
    Etol::Number
    Stol::Number
    solver::AbstractAlgorithm
    GCsweep::Bool
    GCsite::Bool
    verbose::Bool
    isdisk::Bool
    function DMRGalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme, N::Int64, Etol::Number, Stol::Number, solver::AbstractAlgorithm,GCsweep::Bool,GCsite::Bool,verbose::Bool,isdisk::Bool=IS_DISK[])
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,N,Etol,Stol,solver,GCsweep,GCsite,verbose,isdisk)
    end
end

mutable struct TDVPalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    trunc::TruncationScheme
    τ::Number
    tol::Number
    solver::AbstractAlgorithm
    GCsweep::Bool
    GCsite::Bool
    verbose::Bool
    isdisk::Bool
    function TDVPalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme, τ::Number, tol::Number, solver::AbstractAlgorithm,GCsweep::Bool,GCsite::Bool,verbose::Bool,isdisk::Bool=IS_DISK[])
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,τ,tol,solver,GCsweep,GCsite,verbose,isdisk)
    end
end

struct CBEalgo{Sch,Struc,Tar} <: AbstractAlgorithm where {Sch,Struc,Tar}
    scheme::AbstractScheme
    structure::AbstractStructure
    target::Int64
    D::Int64 
    # ϵ::Number
    CBEalgo(alg::CBEalgo, scheme::AbstractScheme) = new{typeof(scheme), typeof(alg.structure), alg.target}(scheme, alg.structure, alg.target, alg.D)
    CBEalgo(scheme::AbstractScheme,structure::AbstractStructure,target::Int64,D::Int64) = new{typeof(scheme),typeof(structure),target}(scheme,structure,target,D)
    CBEalgo(alg::CBEalgo,structure::AbstractStructure) = new{typeof(alg.scheme),typeof(structure),alg.target}(alg.scheme,structure,alg.target,alg.D)
    CBEalgo(alg::CBEalgo,structure::AbstractStructure,target::Int64) = new{typeof(alg.scheme), typeof(structure), target}(alg.scheme, structure, target, alg.D)
end

mutable struct Algebraalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    trunc::TruncationScheme
    N::Int64
    tol::Number
    verbose::Bool
    isdisk::Bool
    function Algebraalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme,N::Int64, tol::Number,verbose::Bool,isdisk::Bool=IS_DISK[])
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,N,tol,verbose,isdisk)
    end
end


mutable struct XTRGalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    N::Int64
    H::Union{SparseMPO,Nothing}
    isdisk::Bool
    function XTRGalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, N::Int64, H::Union{SparseMPO,Nothing} = nothing,isdisk::Bool=IS_DISK[])
        alg.isdisk = isdisk
        new{typeof(scheme),typeof(alg)}(scheme,alg,N,H,isdisk)
    end
end

mutable struct Myhillalgo <: AbstractAlgorithm
    nodes::Vector{Vector{Vector{Int64}}}    # layers → groups → tunnel indices
    weight::Type                            # edge weight type
    Myhillalgo(weight::Type=Number) = new(Vector{Vector{Int64}}[], weight)
end

mutable struct LanczosAlgorithm <: AbstractAlgorithm
    maxdim::Int64
    tol::Float64
    North::Int64 
    isdisk::Bool
    verbose::Bool
    showtimes::Int64
    LanczosAlgorithm(N::Int64, tol::Float64 = 1e-12, North::Int64 = 2, isdisk::Bool = IS_DISK[], verbose::Bool = true, showtimes::Int64 = 10) = new(N,tol,North,isdisk,verbose,showtimes)
end
