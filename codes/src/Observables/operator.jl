mutable struct ObservableOperator{R₁,R₂} <: AbstractLocalOperator{R₁,R₂}
    A::Union{Nothing,AbstractTensorMap}
    name::Union{Nothing,String}
    site::Int64
    isstring::Bool
    EnvL::Union{LeftEnvironmentTensor,Nothing}
    EnvR::Union{RightEnvironmentTensor,Nothing}
    leftdata::Union{Dict,Nothing}
    rightdata::Union{Dict,Nothing}
    lock::ReentrantLock
    # 从旧式 AbstractLocalOperator 拆包
    ObservableOperator{R₁,R₂}(A::ObservableOperator{R₁,R₂}) where {R₁,R₂} = new{R₁,R₂}(A.A,A.name,A.site,A.isstring,A.EnvL,A.EnvR,A.leftdata,A.rightdata,ReentrantLock())
    ObservableOperator(A::AbstractLocalOperator{R₁,R₂}, isstring::Bool) where {R₁,R₂} = new{R₁,R₂}(A.A, A.name, A.site, isstring, nothing, nothing, nothing, nothing, ReentrantLock())
    # 直接创建：和张量对齐
    ObservableOperator(A::AbstractTensorMap, name::String, site::Int64, isstring::Bool = false) = new{length(codomain(A)),length(domain(A))}(A, name, site, isstring, nothing, nothing, nothing, nothing, ReentrantLock())
    # ObservableOperator{R₁,R₂}(A::Union{Nothing,AbstractTensorMap}, isstring::Bool) where {R₁,R₂} = new{R₁,R₂}(A, A.name, A.site, isstring, nothing, nothing, nothing, nothing, ReentrantLock())
end

trivial(A::ObservableOperator) = ObservableOperator(A.A, A.name, A.site, true)
ObservableOperator(i::Int64) = ObservableOperator(IdentityOperator(i), true)


mutable struct CompositeObservableOperator{N} <: AbstractLocalOperator{0,0}
    A::NTuple{N,AbstractLocalOperator}
    isstring::NTuple{N,Bool}
    EnvL::Union{LeftEnvironmentTensor,Nothing}
    EnvR::Union{RightEnvironmentTensor,Nothing}
    leftdata::Union{Dict,Nothing}
    rightdata::Union{Dict,Nothing}
    lock::ReentrantLock
    CompositeObservableOperator{N}(A::CompositeObservableOperator) where N = new{N}(A.A, A.isstring, A.EnvL, A.EnvR, A.leftdata, A.rightdata, ReentrantLock())
    CompositeObservableOperator(A::NTuple{N,AbstractLocalOperator}, isstring::NTuple{N,Bool}) where N = new{N}(A, isstring, nothing, nothing, nothing, nothing,ReentrantLock())
    CompositeObservableOperator(A::Vector{<:ObservableOperator}) = CompositeObservableOperator(NTuple{length(A),AbstractLocalOperator}([isnothing(a.A) ? IdentityOperator(a.site) : LocalOperator(a.A, a.name, a.site) for a in A]), Tuple([a.isstring for a in A]))
    CompositeObservableOperator{N}(i::Int64) where N = CompositeObservableOperator([ObservableOperator(i) for _ in 1:N])
    CompositeObservableOperator{N}(A::NTuple{N,AbstractLocalOperator}, isstring::NTuple{N,Bool}) where N = new{N}(A, isstring, nothing, nothing, nothing, nothing,ReentrantLock())
end


function composite(A::ObservableOperator,B::ObservableOperator)
    @assert isdefault(A) && isdefault(B)
    LA = LocalOperator(A.A, A.name, A.site)
    LB = LocalOperator(B.A, B.name, B.site)
    return CompositeObservableOperator((LA, LB), (A.isstring, B.isstring))
end

Base.isequal(A::T, B::T) where T <: Union{ObservableOperator,CompositeObservableOperator}= (isequal(A.A,B.A) && isequal(A.isstring,B.isstring))
# Base.show(io::IO,A::T) where T <: Union{ObservableOperator,CompositeObservableOperator} = show(io,A.A)
Base.copy(A::T) where T <: Union{ObservableOperator,CompositeObservableOperator}= T(A)
Base.hash(A::ObservableOperator, h::UInt) = hash(A.A, hash(A.isstring, hash(A.name, hash(A.site, h))))
Base.hash(A::CompositeObservableOperator, h::UInt) = hash(A.A, hash(A.isstring, h))
function Base.show(io::IO,A::ObservableOperator)
    print(io,"$(isnothing(A.name) ? "I" : A.name)$(String(collect("$(A.site)") .+ 8272))")
end
function Base.show(io::IO,A::CompositeObservableOperator{N}) where N
    print(io,"(")
    for i in 1:N
        print(io,"$(A.A[i]),")
    end
    print(io,")")
end
TensorKit.norm(A::ObservableOperator) = norm(A.A)

Base.:*(A::Number, B::ObservableOperator) = ObservableOperator(A * B.A, B.name, B.site)
Base.:+(A::ObservableOperator, B::ObservableOperator) = (@assert !A.isstring && !B.isstring "+ is not designed for string operator" ; @assert A.site == B.site; ObservableOperator(A.A + B.A, string(A.name,"+",B.name), A.site, false))
Base.:-(A::ObservableOperator, B::ObservableOperator) = (@assert !A.isstring && !B.isstring "- is not designed for string operator" ; @assert A.site == B.site; ObservableOperator(A.A - B.A, string(A.name,"-",B.name), A.site, false))
Base.:*(A::ObservableOperator, B::ObservableOperator) = (@assert !A.isstring && !B.isstring "* is not designed for string operator" ; @assert A.site == B.site; ObservableOperator(A.A * B.A, string(A.name,"*",B.name), A.site, false))
