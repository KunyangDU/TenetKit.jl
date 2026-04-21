

mutable struct InteractionTreeLeave{N, S<:Number}
    A::NTuple{N,AbstractTensorMap}
    site::NTuple{N,Int64}
    name::NTuple{N,String}
    fermionic::NTuple{N,Bool}
    strength::S
    Z::Union{Nothing,AbstractTensorMap}

    function InteractionTreeLeave(A::NTuple{N,AbstractTensorMap},
            site::NTuple{N,Int64},
            name::NTuple{N,String},
            fermionic::NTuple{N,Bool},
            strength::S,
            Z::Union{Nothing,AbstractTensorMap} = nothing) where {N, S<:Number}
            @assert issorted(site) "sites not sorted!"
        return new{N,S}(A,site,name,fermionic,strength,Z)
    end

    function InteractionTreeLeave(A::AbstractTensorMap,
            site::Int64,
            name::String,
            fermionic::Bool,
            strength::S,
            Z::Union{Nothing,AbstractTensorMap} = nothing) where {S<:Number}
        return new{1,S}((A,),(site,),(name,),(fermionic,),strength,Z)
    end

    function InteractionTreeLeave(
            A::Vector,
            site::Vector,
            name::Vector,
            fermionic::Vector,
            strength::S,
            Z::Union{Nothing,AbstractTensorMap} = nothing) where {S<:Number}
        return new{length(A),S}(Tuple(A),Tuple(site),Tuple(name),Tuple(fermionic),strength,Z)
    end
end

replace!(::Nothing, y::InteractionTreeLeave) = y
function replace!(x::InteractionTreeLeave, y::InteractionTreeLeave)
    for name in fieldnames(y)
        setproperty!(x,name,getproperty(y,name))
    end
    return x
end

function Base.show(::IO, x::InteractionTreeLeave)
    println("name = $(x.name)")
    println("site = $(x.site)")
    println("fermionic = $(x.fermionic)")
    println("strength = $(x.strength)")
    println("Z = $(x.Z)")
end


mutable struct LocalOperator{R₁,R₂,S<:Number} <: AbstractLocalOperator{R₁,R₂}
    A::Union{Nothing,AbstractTensorMap}
    name::String
    site::Int64
    strength::S

    function LocalOperator(A::AbstractTensorMap, name::String, site::Int64)
        return new{length(codomain(A)),length(domain(A)),Float64}(A, name, site, NaN)
    end

    function LocalOperator(A::AbstractTensorMap, name::String, site::Int64, strength::S) where {S<:Number}
        return new{length(codomain(A)),length(domain(A)),S}(A, name, site, strength)
    end

    function LocalOperator(Leave::InteractionTreeLeave{1})
        A  = Leave.A[1]
        st = Leave.strength
        return new{length(codomain(A)),length(domain(A)),typeof(st)}(A, Leave.name[1], Leave.site[1], st)
    end
end

function Base.show(io::IO,A::LocalOperator)
    print(io,"$(A.name)$(String(collect("$(A.site)") .+ 8272))")
    if !isnan(A.strength)
        print(io, "($(A.strength))")
   end
end


mutable struct IdentityOperator{R, S<:Number} <: AbstractLocalOperator{0,0}
    A::Union{Nothing, AbstractTensorMap}
    site::Int64
    strength::S
    name::Union{Nothing,String,Tuple}
    function IdentityOperator(A::AbstractTensorMap, site::Int64)
        return new{length(codomain(A)), Float64}(nothing, site, NaN, nothing)
    end
    function IdentityOperator(site::Int64, strength::S = NaN) where {S<:Number}
        return new{0, S}(nothing, site, strength, nothing)
    end

    function IdentityOperator(site::Int64, name::Union{String,Tuple})
        return new{0, Float64}(nothing, site, NaN, name)
    end

    function IdentityOperator(site::Int64)
        return new{0, Float64}(nothing, site, NaN, nothing)
    end

    function IdentityOperator(A::LocalOperator)
        A′ = getIdTensor(A)
        return new{length(codomain(A′)), Float64}(nothing, deepcopy(A.site), NaN, nothing)
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
    return A.A ≈ B.A
end

function getIdTensor(A::AbstractLocalOperator)
    space = codomain(A.A)[1]
    return TensorMap(diagm(ones(dim(space))),space,space)
end

trivial(A::AbstractLocalOperator) = IdentityOperator(A)

