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

Base.isequal(A::ObservableOperator, B::ObservableOperator) = isequal(A.A,B.A) && isequal(A.isstring,B.isstring)
Base.show(io::IO,A::ObservableOperator) = show(io,A.A)
trivial(A::ObservableOperator) = ObservableOperator(trivial(A.A), true)

ObservableOperator(i::Int64) = ObservableOperator(IdentityOperator(i),true)
