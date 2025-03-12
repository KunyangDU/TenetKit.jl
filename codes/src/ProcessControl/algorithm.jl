struct NoAlgorithm <: AbstractAlgorithm NoAlgorithm() = new() end
# struct noalgorithm <: NoAlgorithm noalgorithm() = new() end


mutable struct Krylovalgo <: SolverAlgo
    Alg::KrylovKit.KrylovAlgorithm
    Krylovalgo(Alg::KrylovKit.KrylovAlgorithm) = new(Alg)
end

mutable struct SETTNalgo{Sch} <: AbstractAlgorithm where {Sch}
    scheme::AbstractScheme
    N::Int64
    D::Int64
    tol::Number
    function SETTNalgo(scheme::AbstractScheme, N::Int64, D::Int64, tol::Number)
        new{typeof(scheme)}(scheme,N,D,tol)
    end
end

mutable struct DMRGalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    trunc::TruncationScheme
    N::Int64
    tol::Number
    solver::SolverAlgo
    function DMRGalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme, N::Int64, tol::Number, solver::SolverAlgo)
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,N,tol,solver)
    end
end

mutable struct TDVPalgo{Sch,Alg} <: AbstractAlgorithm where {Sch,Alg}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    trunc::TruncationScheme
    τ::Number 
    tol::Number
    solver::SolverAlgo
    function TDVPalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, trunc::TruncationScheme, τ::Number, tol::Number, solver::SolverAlgo)
        new{typeof(scheme),typeof(alg)}(scheme,alg,trunc,τ,tol,solver)
    end
end

mutable struct CBEalgo{Sch} <: AbstractAlgorithm where {Sch}
    scheme::AbstractScheme
    λ::Number
    D::Int64 
    ϵ::Number
    CBEalgo(scheme::AbstractScheme,λ::Number,D::Int64,ϵ::Number) = new{typeof(scheme)}(scheme,λ,D,ϵ)
end

#= ========================= =#

# function merge!(A::Krylovalgo,B::Krylovalgo)
#     @assert A.A == B.A "KrylovAlgorithm mistmatch" 
# end
# function merge(A::Krylovalgo,B::Krylovalgo)
#     @assert A.A == B.A "KrylovAlgorithm mistmatch" 
#     return A
# end



