
mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator{R₁,R₂}
    A::Union{Nothing,AbstractTensorMap}
    name::String
    site::Int64
    strength::Union{Nothing,Number}

    function LocalOperator(A::AbstractTensorMap, name::String, site::Int64)
        return new{length(codomain(A)),length(domain(A))}(A, name, site, NaN)
    end

    function LocalOperator(A::AbstractTensorMap, name::String, site::Int64, strength::Number)
        return new{length(codomain(A)),length(domain(A))}(A, name, site, strength)
    end
end

function Base.show(io::IO,A::LocalOperator)
    print(io,"$(A.name)$(String(collect("$(A.site)") .+ 8272))")
    if !isnan(A.strength)
        print(io, "($(A.strength))")
   end
end


mutable struct IdentityOperator{R} <: AbstractLocalOperator{0,0}
    A::Union{Nothing, AbstractTensorMap}
    site::Int64
    strength::Number 
    name::Union{Nothing,String,Tuple}
    function IdentityOperator(A::AbstractTensorMap,site::Int64)
        return new{length(codomain(A))}(nothing, site, NaN ,nothing)
   end
    function IdentityOperator(site::Int64, strength::Number = NaN)
         return new{1}(nothing, site, strength,nothing)
    end

    function IdentityOperator(site::Int64, name::Union{String,Tuple})
        return new{1}(nothing, site, NaN ,name)
    end

    function IdentityOperator(site::Int64)
        return new{1}(nothing, site, NaN ,nothing)
    end

    function IdentityOperator(A::LocalOperator)
        A′ = getIdTensor(A)
        return new{length(codomain(A′))}(nothing, A.site, NaN ,nothing)
    end

    function IdentityOperator(A::IdentityOperator)
        return A
    end

    IdentityOperator() = IdentityOperator(0)
end

function Base.show(io::IO,A::IdentityOperator)
    print(io,"I$(String(collect("$(A.site)") .+ 8272))")
    if !isnan(A.strength)
        print(io, "($(A.strength))")
    end
    if !isnothing(A.name)
        print(io, "{$(A.name)}")
    end
end



# Base.isequal(::AbstractLocalOperator, ::AbstractLocalOperator) = false
Base.isequal(A::IdentityOperator, B::IdentityOperator) = (A.site == B.site && A.name == B.name)
function Base.isequal(A::LocalOperator, B::LocalOperator)
    A.name ≠ B.name && return false
    A.site ≠ B.site && return false
    # if both NaN -> True
    # else check strength
    # repeat interaction added without merging
    !(isnan(A.strength) && isnan(B.strength)) && !(A.strength ≈ B.strength) && return false
    # return A.A ≈ B.A
    return isequal(A.A,B.A)
end

function getIdTensor(A::AbstractLocalOperator)
    space = codomain(A.A)[1]
    return TensorMap(diagm(ones(dim(space))),space,space)
end

trivial(A::AbstractLocalOperator) = IdentityOperator(A)

