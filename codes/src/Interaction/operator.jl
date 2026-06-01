
mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator{R₁,R₂}
    A::AbstractTensorMap
    name::String
    site::Int64

    LocalOperator(A::AbstractTensorMap, name::String, site::Int64) = new{length(codomain(A)),length(domain(A))}(A, name, site)
    LocalOperator{R₁,R₂}(A::AbstractTensorMap, name::String, site::Int64) where {R₁,R₂} = new{R₁,R₂}(A, name, site)
end

function Base.show(io::IO,A::LocalOperator)
    print(io,"$(A.name)$(String(collect("$(A.site)") .+ 8272))")
end


mutable struct IdentityOperator{R} <: AbstractLocalOperator{0,0}
    A::Union{Nothing,AbstractTensorMap}
    name::Union{Nothing,String}
    site::Int64
    function IdentityOperator(A::AbstractTensorMap,site::Int64)
        return new{length(codomain(A))}(nothing, nothing, site)
   end
    function IdentityOperator(site::Int64, name::String)
        return new{1}(nothing, name, site)
    end

    function IdentityOperator(site::Int64)
        return new{1}(nothing, nothing, site)
    end

    function IdentityOperator(A::LocalOperator)
        A′ = getIdTensor(A)
        return new{length(codomain(A′))}(nothing, nothing, A.site)
    end

    function IdentityOperator(A::IdentityOperator)
        return A
    end

    IdentityOperator() = IdentityOperator(0)
    IdentityOperator{R}(::Nothing,::Nothing,site::Int64) where R = new{R}(nothing,nothing,site)
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

Base.copy(A::T) where T <: Union{LocalOperator,IdentityOperator} = T(A.A, A.name, A.site)

function getIdTensor(A::AbstractLocalOperator)
    space = codomain(A.A)[1]
    # return TensorMap(diagm(ones(dim(space))),space,space)
    return isometry(space,space)
end

trivial(A::T) where T<: Union{LocalOperator,IdentityOperator} = IdentityOperator(A)

TensorKit.norm(A::LocalOperator) = norm(A.A)

Base.:*(A::Number, B::LocalOperator) = LocalOperator(A * B.A, B.name, B.site)
Base.:*(A::LocalOperator, B::LocalOperator) = (@assert A.site == B.site; LocalOperator(A.A * B.A, string(A.name,B.name), A.site))
Base.:+(A::LocalOperator, B::LocalOperator) = (@assert A.site == B.site; LocalOperator(A.A + B.A, string(A.name, "+", B.name), A.site))
Base.:-(A::LocalOperator, B::LocalOperator) = (@assert A.site == B.site; LocalOperator(A.A - B.A, string(A.name, "-", B.name), A.site))
