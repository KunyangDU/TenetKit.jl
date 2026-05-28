
mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator{R₁,R₂}
    A::Union{Nothing,AbstractTensorMap}
    name::String
    site::Int64

    LocalOperator(A::AbstractTensorMap, name::String, site::Int64) = new{length(codomain(A)),length(domain(A))}(A, name, site)
end

function Base.show(io::IO,A::LocalOperator)
    print(io,"$(A.name)$(String(collect("$(A.site)") .+ 8272))")
end


mutable struct IdentityOperator{R} <: AbstractLocalOperator{0,0}
    A::Union{Nothing, AbstractTensorMap}
    site::Int64
    name::Union{Nothing,String,Tuple}
    function IdentityOperator(A::AbstractTensorMap,site::Int64)
        return new{length(codomain(A))}(nothing, site, nothing)
   end
    function IdentityOperator(site::Int64, name::Union{String,Tuple})
        return new{1}(nothing, site, name)
    end

    function IdentityOperator(site::Int64)
        return new{1}(nothing, site, nothing)
    end

    function IdentityOperator(A::LocalOperator)
        A′ = getIdTensor(A)
        return new{length(codomain(A′))}(nothing, A.site, nothing)
    end

    function IdentityOperator(A::IdentityOperator)
        return A
    end

    IdentityOperator() = IdentityOperator(0)
end

function Base.show(io::IO,A::IdentityOperator)
    print(io,"I$(String(collect("$(A.site)") .+ 8272))")
    if !isnothing(A.name)
        print(io, "{$(A.name)}")
    end
end



# Base.isequal(::AbstractLocalOperator, ::AbstractLocalOperator) = false
Base.isequal(A::IdentityOperator, B::IdentityOperator) = (A.site == B.site && A.name == B.name)
function Base.isequal(A::LocalOperator, B::LocalOperator)
    A.name ≠ B.name && return false
    A.site ≠ B.site && return false
    return A.A ≈ B.A
end

function getIdTensor(A::AbstractLocalOperator)
    space = codomain(A.A)[1]
    # return TensorMap(diagm(ones(dim(space))),space,space)
    return isometry(space,space)
end

trivial(A::T) where T<: Union{LocalOperator,IdentityOperator} = IdentityOperator(A)

