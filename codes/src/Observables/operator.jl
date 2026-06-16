mutable struct CompositeLocalOperator{N} <: AbstractLocalOperator{0,0}
    A::NTuple{N,AbstractLocalOperator}
    isstring::NTuple{N,Bool}
    CompositeLocalOperator(A::NTuple{N,AbstractLocalOperator}, isstring::NTuple{N,Bool}) where N = new{N}(A, isstring)
    CompositeLocalOperator{N}(A::CompositeLocalOperator{N}) where N = new{N}(A.A, A.isstring)
    CompositeLocalOperator(A::Vector{<:AbstractLocalOperator}) = CompositeLocalOperator(NTuple{length(A),AbstractLocalOperator}([isnothing(a.A) ? IdentityOperator(a.site) : LocalOperator(a.A, a.name, a.site) for a in A]), Tuple([a.isstring for a in A]))
    CompositeLocalOperator{N}(i::Int64) where N = CompositeLocalOperator([LocalOperator(i) for _ in 1:N])
end

function composite(A::LocalOperator, B::LocalOperator)
    # @assert isdefault(A) && isdefault(B)
    LA = LocalOperator(A.A, A.name, A.site)
    LB = LocalOperator(B.A, B.name, B.site)
    return CompositeLocalOperator((LA, LB), (A.isstring, B.isstring))
end

Base.isequal(A::T, B::T) where T <: CompositeLocalOperator = (isequal(A.A, B.A) && isequal(A.isstring, B.isstring))
Base.copy(A::T) where T <: CompositeLocalOperator = T(A)
Base.hash(A::CompositeLocalOperator, h::UInt) = hash(A.A, hash(A.isstring, h))
