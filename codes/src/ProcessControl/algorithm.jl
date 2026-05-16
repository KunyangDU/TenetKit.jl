struct NoAlgorithm <: AbstractAlgorithm NoAlgorithm() = new() end
# struct noalgorithm <: NoAlgorithm noalgorithm() = new() end


struct Krylovalgo <: SolverAlgo
    Alg::KrylovKit.KrylovAlgorithm
    Krylovalgo(Alg::KrylovKit.KrylovAlgorithm) = new(Alg)
end

struct Chebyshev <: SolverAlgo
    tol::Float64
    maxiter::Int
    function Chebyshev(; tol::Float64=1e-10, maxiter::Int=200)
        new(tol, maxiter)
    end
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
    isdisk::Bool
    function DMRGalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme, N::Int64, Etol::Number, Stol::Number, solver::AbstractAlgorithm,GCsweep::Bool,GCsite::Bool,isdisk::Bool=false)
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,N,Etol,Stol,solver,GCsweep,GCsite,isdisk)
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
    isdisk::Bool
    function TDVPalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme, τ::Number, tol::Number, solver::AbstractAlgorithm,GCsweep::Bool,GCsite::Bool,isdisk::Bool=false)
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,τ,tol,solver,GCsweep,GCsite,isdisk)
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

struct Algebraalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    trunc::TruncationScheme
    N::Int64
    tol::Number
    isdisk::Bool
    function Algebraalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme,N::Int64, tol::Number,isdisk::Bool=false)
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,N,tol,isdisk)
    end
end

mutable struct LKANalgo{N,M} <: AbstractAlgorithm where {N,M}
    order::Int64
    mode::Symbol
    filepath::String
    tailname::String
    solver::SolverAlgo
    scale::Number
    algo::SolverAlgo
    count::Int64
    function LKANalgo(order::Int64, mode::Symbol, filepath::String, tailname::String, solver::SolverAlgo, scale::Number = 1., algo::SolverAlgo = mode == :dmrg ? DMRGDefaultLanczos : HamiltonianBoundDefaultLanczos, count::Int64 = 0)
        new{order,mode}(order, mode, filepath, tailname, solver, scale, algo, count)
    end
end

#= ========================= =#

# function merge!(A::Krylovalgo,B::Krylovalgo)
#     @assert A.A == B.A "KrylovAlgorithm mistmatch" 
# end
# function merge(A::Krylovalgo,B::Krylovalgo)
#     @assert A.A == B.A "KrylovAlgorithm mistmatch" 
#     return A
# end


mutable struct XTRGalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    N::Int64
    H::Union{SparseMPO,Nothing}
    isdisk::Bool
    function XTRGalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, N::Int64, H::Union{SparseMPO,Nothing} = nothing,isdisk::Bool=false)
        new{typeof(scheme),typeof(alg)}(scheme,alg,N,H,isdisk)
    end
end

