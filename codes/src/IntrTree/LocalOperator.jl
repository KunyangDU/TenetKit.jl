
mutable struct IdentityOperator{R} <: AbstractLocalOperator
    A::Union{Nothing, AbstractTensorMap}
    site::Int64
    strength::Number 
    name::Union{Nothing,String,Tuple}
    function IdentityOperator(A::AbstractTensorMap,site::Int64)
        return new{length(codomain(A))}(A, site, NaN ,nothing)
   end
    function IdentityOperator(site::Int64, strength::Number = NaN)
         return new{0}(nothing, site, strength,nothing)
    end

    function IdentityOperator(site::Int64, name::Union{String,Tuple})
        return new{0}(nothing, site, NaN ,name)
    end

    function IdentityOperator(site::Int64)
        return new{0}(nothing, site, NaN ,nothing)
    end

    IdentityOperator() = IdentityOperator(0)
end

function Base.show(io::IO,Opr::IdentityOperator)
    print(io,"I$(String(collect("$(Opr.site)") .+ 8272))")
    if !isnan(Opr.strength)
        print(io, "($(Opr.strength))")
    end
    if !isnothing(Opr.name)
        print(io, "{$(Opr.name)}")
    end
end

mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator
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

function Base.show(io::IO,Opr::LocalOperator)
    print(io,"$(Opr.name)$(String(collect("$(Opr.site)") .+ 8272))")
    if !isnan(Opr.strength)
        print(io, "($(Opr.strength))")
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
    return A.A ≈ B.A
end

function getIdTensor(Opr::AbstractLocalOperator)
    space = codomain(Opr.A)[1]
    return TensorMap(diagm(ones(dim(space))),space,space)
end

