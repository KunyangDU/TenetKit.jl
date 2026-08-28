
mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator{R₁,R₂}
    A::AbstractTensorMap
    name::String
    site::Int64
    isstring::Bool

    LocalOperator(A::AbstractTensorMap, name::String, site::Int64, isstring::Bool = false) = new{numout(A),numin(A)}(A, name, site, isstring)
    LocalOperator{R₁,R₂}(A::AbstractTensorMap, name::String, site::Int64, isstring::Bool = false) where {R₁,R₂} = new{R₁,R₂}(A, name, site, isstring)
    LocalOperator{R₁′,R₂′}(mapping::Function, A::LocalOperator{R₁′,R₂′}) where {R₁′,R₂′} = new{R₁′,R₂′}(mapping(A.A),A.name,A.site,A.isstring)
end

function Base.show(io::IO,A::LocalOperator)
    print(io,"$(A.name)$(String(collect("$(A.site)") .+ 8272))")
end


mutable struct IdentityOperator{R} <: AbstractLocalOperator{0,0}
    A::Union{Nothing,AbstractTensorMap}
    name::Union{Nothing,String}
    site::Int64
    isstring::Bool
    IdentityOperator(A::AbstractTensorMap,site::Int64) = new{numout(A)}(A, nothing, site, true)
    IdentityOperator(site::Int64, isstring::Bool) = new{1}(nothing, nothing, site, isstring)
    IdentityOperator(site::Int64, name::String, isstring::Bool = true) = new{1}(nothing, name, site, isstring)
    IdentityOperator(site::Int64) = new{1}(nothing, nothing, site, true)
    IdentityOperator(S::ElementarySpace, site::Int64) = new{1}(isometry(S,S), nothing, site, true)
    IdentityOperator(::Nothing, site::Int64) = new{1}(nothing, nothing, site, true)

    function IdentityOperator(A::LocalOperator)
        A′ = getIdTensor(A)
        return new{numout(A′)}(A′, nothing, A.site, A.isstring)
    end

    IdentityOperator(A::IdentityOperator) = A
    IdentityOperator() = IdentityOperator(0)
    IdentityOperator{R}(::Nothing,::Nothing,site::Int64,isstring::Bool = true) where R = new{R}(nothing,nothing,site,isstring)
    IdentityOperator{R}(::Function,A::IdentityOperator{R}) where R = new{R}(A.A,A.name,A.site,A.isstring)
end

function Base.show(io::IO,A::IdentityOperator)
    print(io,"I$(String(collect("$(A.site)") .+ 8272))")
    if !isnothing(A.name)
        print(io, "{$(A.name)}")
    end
end



# Base.isequal(::AbstractLocalOperator, ::AbstractLocalOperator) = false
Base.isequal(A::IdentityOperator, B::IdentityOperator) = (A.site == B.site && A.name == B.name && A.isstring == B.isstring)
Base.isequal(A::LocalOperator, B::LocalOperator) = (space(A.A) == space(B.A) && A.A ≈ B.A && A.name == B.name && A.site == B.site && A.isstring == B.isstring)

Base.copy(A::T) where T <: Union{LocalOperator,IdentityOperator} = T(A.A, A.name, A.site, A.isstring)
Base.hash(A::LocalOperator, h::UInt) = hash(A.A, hash(A.name, hash(A.site, hash(A.isstring, h))))
Base.hash(A::IdentityOperator, h::UInt) = hash(A.site, hash(A.name, hash(A.isstring, h)))

function getIdTensor(A::AbstractLocalOperator)
    space = codomain(A.A)[1]
    # return TensorMap(diagm(ones(dim(space))),space,space)
    return isometry(space,space)
end

LocalOperator(i::Int64) = IdentityOperator(i, true)
trivial(A::T) where T<: Union{LocalOperator,IdentityOperator} = IdentityOperator(A)

TensorKit.norm(A::LocalOperator) = norm(A.A)

Base.:*(A::Number, B::LocalOperator) = LocalOperator(A * B.A, B.name, B.site)
Base.:*(A::LocalOperator, B::LocalOperator) = (@assert A.site == B.site; LocalOperator(A.A * B.A, string(A.name,B.name), A.site))
Base.:+(A::LocalOperator, B::LocalOperator) = (@assert A.site == B.site; LocalOperator(A.A + B.A, string(A.name, "+", B.name), A.site))
Base.:-(A::LocalOperator, B::LocalOperator) = (@assert A.site == B.site; LocalOperator(A.A - B.A, string(A.name, "-", B.name), A.site))

trivial(::GradedSpace{I, D}) where {I, D} = GradedSpace{I,D}(TensorKit.SortedVectorDict(one(I) => 1), false)
trivial(::ComplexSpace) = ℂ^1