struct NoAlgorithm <: AbstractAlgorithm NoAlgorithm() = new() end
# struct noalgorithm <: NoAlgorithm noalgorithm() = new() end


mutable struct Krylovalgo <: SolverAlgo
    Alg::KrylovKit.KrylovAlgorithm
    Krylovalgo(Alg::KrylovKit.KrylovAlgorithm) = new(Alg)
end

mutable struct DMRGalgo{Sch,Alg,Dim,Tol} <: AbstractAlgorithm where {Sch,Alg,Dim,Tol}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    D::Int64
    ϵ::Number
    N::Int64
    tol::Number
    solver::SolverAlgo
    function DMRGalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, D::Int64, ϵ::Number,N::Int64,tol::Number, solver::SolverAlgo)
        new{scheme,typeof(alg),D,ϵ}(scheme,alg,D,ϵ,N,tol,solver)
    end
end

mutable struct TDVPalgo{Sch,Alg,Dim,Tol} <: AbstractAlgorithm where {Sch,Alg,Dim,Tol}
    scheme::AbstractScheme
    alg::AbstractAlgorithm
    D::Int64
    ϵ::Number
    τ::Number 
    tol::Number
    solver::SolverAlgo
    function TDVPalgo(scheme::AbstractScheme, alg::AbstractAlgorithm, D::Int64, ϵ::Number,τ::Number,tol::Number, solver::SolverAlgo)
        new{scheme,typeof(alg),D,ϵ}(scheme,alg,D,ϵ,τ,tol,solver)
    end
end

mutable struct CBEalgo{Sch,Rat} <: AbstractAlgorithm where {Sch,Rat}
    scheme::AbstractScheme
    λ::Number
    CBEalgo(scheme::AbstractScheme,λ::Number) = new{scheme,λ}(scheme,λ)
end

#= ========================= =#

# function merge!(A::Krylovalgo,B::Krylovalgo)
#     @assert A.A == B.A "KrylovAlgorithm mistmatch" 
# end
# function merge(A::Krylovalgo,B::Krylovalgo)
#     @assert A.A == B.A "KrylovAlgorithm mistmatch" 
#     return A
# end



