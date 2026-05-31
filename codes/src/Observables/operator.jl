mutable struct ObservableOperator{R₁,R₂} <: AbstractLocalOperator{R₁,R₂}
    A::AbstractLocalOperator{R₁,R₂}
    isstring::Bool
    EnvL::Union{LeftEnvironmentTensor,Nothing}
    EnvR::Union{RightEnvironmentTensor,Nothing}
    leftdata::Union{Dict,Nothing}
    rightdata::Union{Dict,Nothing}
    lock::ReentrantLock
    ObservableOperator(A::AbstractLocalOperator{R₁,R₂}, isstring::Bool) where {R₁,R₂} = new{R₁,R₂}(A,isstring,nothing,nothing,nothing,nothing,ReentrantLock())
    ObservableOperator(A::AbstractTensorMap, name::String, site::Int64, isstring::Bool = false) = new{length(codomain(A)),length(domain(A))}(LocalOperator(A, name, site),isstring,nothing,nothing,nothing,nothing,ReentrantLock())
end

trivial(A::ObservableOperator) = ObservableOperator(trivial(A.A), true)
ObservableOperator(i::Int64) = ObservableOperator(IdentityOperator(i),true)


mutable struct CompositeObservableOperator{N} <: AbstractLocalOperator{0,0}
    A::NTuple{N,AbstractLocalOperator}
    isstring::NTuple{N,Bool}
    EnvL::Union{LeftEnvironmentTensor,Nothing}
    EnvR::Union{RightEnvironmentTensor,Nothing}
    leftdata::Union{Dict,Nothing}
    rightdata::Union{Dict,Nothing}
    lock::ReentrantLock
    CompositeObservableOperator(A::NTuple{N,AbstractLocalOperator}, isstring::NTuple{N,Bool}) where N = new{N}(A, isstring, nothing, nothing, nothing, nothing,ReentrantLock())
    CompositeObservableOperator(A::Vector{<:ObservableOperator}) = CompositeObservableOperator(NTuple{length(A),AbstractLocalOperator}([a.A for a in A]), Tuple([a.isstring for a in A]))
    CompositeObservableOperator{N}(i::Int64) where N = CompositeObservableOperator([ObservableOperator(i) for _ in 1:N])
end


function composite(A::ObservableOperator,B::ObservableOperator)
    @assert isdefault(A) && isdefault(B)
    return CompositeObservableOperator((A.A, B.A), (A.isstring , B.isstring))
end

Base.isequal(A::T, B::T) where T <: Union{ObservableOperator,CompositeObservableOperator}= isequal(A.A,B.A) && isequal(A.isstring,B.isstring)
Base.show(io::IO,A::T) where T <: Union{ObservableOperator,CompositeObservableOperator} = show(io,A.A)


