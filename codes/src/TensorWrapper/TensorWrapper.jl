
function TensorKit.scalartype(A::AbstractTensorWrapper)
    return TensorKit.scalartype(A.A)
end

Base.similar(A::AbstractTensorWrapper, ::Type{S}) where {S<:Number} = zerovector(A, S)
function TensorKit.zerovector(A::T, ::Type{S}) where {S<:Number, T<:AbstractTensorWrapper}
    return convert(T, TensorKit.zerovector(A.A, S))
end  
function TensorKit.zerovector!(A::AbstractTensorWrapper) 
    TensorKit.zerovector!(A.A)
    return A
end

TensorKit.inner(A::T,B::T) where T <: AbstractTensorWrapper = inner(A.A,B.A)

Base.convert(::Type{T}, A::AbstractTensorMap) where {T<:AbstractTensorWrapper} = T(A)

function TensorKit.LinearAlgebra.rmul!(A::AbstractTensorWrapper, α::Number)
    TensorKit.LinearAlgebra.rmul!(A.A, α)
    return A
end
function TensorKit.LinearAlgebra.mul!(A::T, B::T, α::Number) where {T<:AbstractTensorWrapper}
    TensorKit.LinearAlgebra.mul!(A.A, B.A, α)
    return A
end

function add!!(A::AbstractTensorWrapper,
    B::AbstractTensorWrapper,
    β::Number = one(scalartype(B)),
    α::Number = one(scalartype(A))
    ) 
    T = promote_type(scalartype(A.A), scalartype(B.A), typeof(α), typeof(β))
    if T <: scalartype(A.A)
         return axpby!(β, B, α, A)
    else
         return α*A + β*B
    end
end

function axpy!(α::Number, A::T, B::T) where {T<:AbstractTensorWrapper}
    # axpy!(α, A.A, B.A)
    B.A = α * A.A + B.A
    return B
end
# axpy!(::Number, ::Nothing, A::AbstractTensorWrapper) = A
# axpy!(α::Number, A::AbstractTensorWrapper, ::Nothing) = α * A
# axpby!(α::Number, ::Nothing, β::Number, A::AbstractTensorWrapper) = rmul!(A, β)
# axpby!(α::Number, A::AbstractTensorWrapper, β::Number, ::Nothing) = axpy!(α, A, nothing)

# axpy!(α::Number, A::AbstractTensorWrapper, ::Nothing) = α * A
# axpby!(::Number, ::Nothing, β::Number, A::AbstractTensorWrapper) = rmul!(A, β)
# axpby!(α::Number, A::AbstractTensorWrapper, β::Number, ::Nothing) = axpy!(α, A, nothing)
function axpby!(α::Number, A::AbstractTensorWrapper, β::Number, B::AbstractTensorWrapper)
    B.A = α * A.A + β * B.A
    return B
end
axpby!(α::Number, A::AbstractTensorWrapper, ::Number, ::Nothing) = α * A
axpby!(::Number, ::Nothing, β::Number, A::AbstractTensorWrapper) = rmul!(A, β)
axpy!(α::Number, A::AbstractTensorWrapper, ::Nothing) = α * A
axpy!(::Number, ::Nothing, B::AbstractTensorWrapper) = B
function axpy!(α::Number, x::AbstractLocalOperator, y::AbstractLocalOperator)
    # @show "check"
    # @assert x.Opri ≈ y.Opri
    @assert x.site == y.site
    # @assert x.name == y.name
    y.strength += α * x.strength
    y.A += x.A
    return y
end

axpy!(::Number,::Nothing,y::AbstractLocalOperator) = y
axpy!(α::Number,x::AbstractLocalOperator,::Nothing) = α * x

add!(A::AbstractTensorWrapper, B::AbstractTensorWrapper) = axpy!(true, B, A)
add!(A::AbstractTensorWrapper, ::Nothing) = A
add!(::Nothing, A::AbstractTensorWrapper) = A

TensorKit.scale!(A::AbstractTensorWrapper, α::Number) = rmul!(A, α)
TensorKit.scale(A::AbstractTensorWrapper, α::Number) = α * A

function scale!!(A::AbstractTensorWrapper, α::S) where {S<:Number}
    T = promote_type(scalartype(A.A), S)
    if T <: scalartype(A)
         return scale!(A, α)
    else
         return scale(A, α)
    end
end

Base.iterate(t::AbstractTensorWrapper) = (t.A,nothing)
Base.iterate(::AbstractTensorWrapper,::Nothing) = nothing
TensorKit.norm(A::AbstractTensorWrapper) = norm(A.A)

showdomain(A::AbstractTensorWrapper) = showdomain(A.A)

Base.isapprox(A::AbstractTensorWrapper,B::AbstractTensorWrapper) = isapprox(A.A , B.A)
TensorKit.space(A::AbstractTensorWrapper) = space(A.A)
TensorKit.space(A::AbstractLocalOperator) = space(A.A)

TensorKit.dims(A::AbstractTensorWrapper) = dims(A.A)

issparse(::T) where T <: Union{DenseMPS,AdjointMPS,DenseMPO,AdjointMPO} = false
issparse(::SparseMPO) = true

_isdisk(obj::T) where T <: Union{DenseMPS,AdjointMPS,DenseMPO,AdjointMPO} = obj.isdisk
_isdisk(::SparseMPO) = false
_isdisk(::RefMPO) = false
Base.size(t::DenseMPOTensor{4}) = map(dim,t.A |> x -> (codomain(x)[2],domain(x)[1]))
Base.length(::DenseMPO{L}) where L = L
Base.length(::AdjointMPO{L}) where L = L
Base.length(::SparseMPO{L}) where L = L
Base.length(::DenseMPS{L}) where {L} = L
Base.length(::AdjointMPS{L}) where {L} = L
Base.length(::RefMPO{L}) where {L} = L

Base.firstindex(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = 1
Base.lastindex(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = lastindex(obj.ts)
Base.size(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = (lastindex(obj),)
Base.axes(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = Base.OneTo(lastindex(obj))

Base.firstindex(obj::RefMPO) = 1
Base.lastindex(obj::RefMPO) = lastindex(obj.ts)
Base.size(obj::RefMPO) = (lastindex(obj),)
Base.axes(obj::RefMPO) = Base.OneTo(lastindex(obj))
Base.size(::SparseMPOTensor{N,M}) where {N,M} = N,M

function normalize!(obj::Union{DenseMPO{L},DenseMPS{L},AdjointMPO{L},AdjointMPS{L}}) where {L}
    @assert (site = obj.center[1]) == obj.center[2]
    return normalize!(obj[site])
end

function TensorKit.norm(obj::Union{DenseMPO{L},DenseMPS{L},AdjointMPO{L},AdjointMPS{L}}) where {L}
    @assert (site = obj.center[1]) == obj.center[2]
    return norm(obj[site])
end

function normalize!(obj::AbstractTensorWrapper)
    tmp = norm(obj.A)
    obj.A = obj.A / tmp
    return tmp
end

Base.:+(A::T, B::T) where T <: AbstractTensorWrapper = T(A.A + B.A)
Base.:+(::Nothing, B::AbstractTensorWrapper) = B
Base.:+(A::AbstractTensorWrapper, ::Nothing) = A
Base.:-(A::T, B::T) where T <: AbstractTensorWrapper = T(A.A - B.A)
Base.:*(A::T,B::T) where T <: AbstractTensorWrapper = T(A.A * B.A)
Base.:*(A::Number,B::T) where T <: AbstractTensorWrapper = T(A * B.A)
Base.:/(A::T,B::Number) where T <: AbstractTensorWrapper = (1/B) * A
function Base.:*(A::Number,B::AbstractLocalOperator)
    B.strength *= A
    return B 
end


scale(t::Tuple{T₁,T₂}) where {T₁ <: AbstractTensorWrapper,T₂ <: Number} = scale((t[1].A,t[2]))
scale!!(t::Tuple{T₁,T₂}) where {T₁ <: AbstractTensorWrapper,T₂ <: Number} = scale!!((t[1].A,t[2]))
scale!!(t::Tuple{T₁,T₂,T₃}) where {T₁ <: AbstractTensorWrapper,T₂ <: AbstractTensorWrapper,T₃ <: Number} = scale!!((t[1].A,t[2].A,t[3]))
zerovector(t::Tuple{T₁,T₂}) where {T₁ <: AbstractTensorWrapper,T₂ <: Number} = zerovector((t[1].A,t[2]))
add!!(t::Tuple{T₁,T₂,T₃,T₄}) where {T₁ <: AbstractTensorWrapper,T₂ <: AbstractTensorWrapper,T₃ <: Number,T₄} = add!!((t[1].A,t[2].A,t[3],t[4]))

TensorKit.codomain(A::AbstractTensorWrapper) = codomain(A.A)
TensorKit.domain(A::AbstractTensorWrapper) = domain(A.A)
Base.eltype(A::AbstractTensorWrapper) = eltype(A.A)
Base.randn(A::T) where T <: AbstractTensorWrapper = T(TensorMap(randn, eltype(A), codomain(A), domain(A)))

Base.copy(A::T) where T <: AbstractTensorWrapper = T(copy(A.A))

rank(A::T) where T <: AbstractTensorWrapper = rank(A.A)

Base.getindex(obj::T, i::Int64) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "deserialize" obj.ts[i]) : obj.ts[i]
Base.getindex(obj::RefMPO, i::Int64) = obj.mapping(obj.ts[i])
Base.getindex(obj::T, stp::UnitRange) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "deserialize" [obj.ts[i] for i in stp]) : [obj.ts[i] for i in stp]
Base.getindex(obj::RefMPO, stp::UnitRange) = obj.mapping.([obj.ts[i] for i in stp])

Base.setindex!(obj::T, val, i::Int64) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "serialize" obj.ts[i] = val) : (obj.ts[i] = val)
Base.setindex!(obj::T, vals, stp::UnitRange) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "serialize" for (i, v) in zip(stp, vals); obj.ts[i] = v; end) : (for (i, v) in zip(stp, vals); obj.ts[i] = v; end)
Base.setindex!(obj::RefMPO, val, i::Int64) = (obj.ts[i] = val)

Base.getindex(obj::T, ::Colon) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? [obj.ts[i] for i in 1:length(obj.ts)] : obj.ts[:]
Base.getindex(obj::RefMPO, ::Colon) = obj.mapping.(obj.ts[:])
Base.setindex!(obj::T, vals, ::Colon) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (for (i, v) in enumerate(vals); obj.ts[i] = v; end) : (obj.ts[:] = vals)

Base.setindex!(obj::T, val, i::Int64) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = (obj.ts[i] = val)
Base.setindex!(obj::T, vals, stp::UnitRange) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = (obj.ts[stp] = vals)
Base.setindex!(obj::RefMPO, val, i::Int64) = (obj.ts[i] = val)

# function Base.:-(A::AbstractMPOTensor, B::AbstractMPOTensor)
#     return A + (-1) * B
# end

# function Base.:-(A::MPSTensor{R₁}, B::MPSTensor{R₂}) where {R₁,R₂}
#     return A + (-1)*B
# end

# function Base.:-(A::CompositeMPSTensor{2, 4}, B::CompositeMPSTensor{2, 4})
#     return A + (-1)*B
# end

# function Base.:-(A::RightCompositeEnvironmentTensor{N₁, R₁}, B::RightCompositeEnvironmentTensor{N₂, R₂}) where {N₁,N₂,R₁,R₂}
#     @assert N₁ == N₂ && R₁ == R₂
#     return RightCompositeEnvironmentTensor(A.A - B.A)
# end

# function Base.:-(A::LeftCompositeEnvironmentTensor{N₁, R₁}, B::LeftCompositeEnvironmentTensor{N₂, R₂}) where {N₁,N₂,R₁,R₂}
#     @assert N₁ == N₂ && R₁ == R₂
#     return LeftCompositeEnvironmentTensor(A.A - B.A)
# end


# function Base.:+(A::LeftCompositeEnvironmentTensor,
#     B::LeftCompositeEnvironmentTensor)
#     return LeftCompositeEnvironmentTensor(A.A + B.A)
# end

# function Base.:+(A::RightCompositeEnvironmentTensor,
#     B::RightCompositeEnvironmentTensor)
#     return RightCompositeEnvironmentTensor(A.A + B.A)
# end

# function Base.:+(A::LeftEnvironmentTensor,
#     B::LeftEnvironmentTensor)
#     return LeftEnvironmentTensor(A.A + B.A)
# end

# function Base.:+(A::RightEnvironmentTensor,
#     B::RightEnvironmentTensor)
#     return RightEnvironmentTensor(A.A + B.A)
# end

# function Base.:+(A::CompositeMPOTensor{N₁, R₁}, B::CompositeMPOTensor{N₂, R₂}) where {N₁, N₂, R₁, R₂}
#     @assert N₁ == N₂ && R₁ == R₂
#     return CompositeMPOTensor(A.A + B.A)
# end

# function Base.:+(A::DenseMPOTensor{R₁}, B::DenseMPOTensor{R₂}) where {R₁, R₂}
#     @assert R₁ == R₂
#     return DenseMPOTensor(A.A + B.A)
# end

# function Base.:+(A::MPSTensor{R₁}, B::MPSTensor{R₂}) where {R₁,R₂}
#     @assert R₁ == R₂
#     return MPSTensor(A.A + B.A)
# end
# function Base.:+(A::CompositeMPSTensor{2, 4}, B::CompositeMPSTensor{2, 4})
#     return CompositeMPSTensor(A.A + B.A)
# end

# function Base.:+(::Nothing,A::DenseMPOTensor)
#     return A
# end

# function Base.:+(A::DenseMPOTensor,::Nothing)
#     return A
# end


# function Base.:*(α::Number, A::CompositeMPOTensor)
#     return  CompositeMPOTensor(α*A.A)
# end

# function Base.:*(α::Number, A::DenseMPOTensor)
#     return  DenseMPOTensor(α*A.A)
# end

# function Base.:*(n::Number, A::MPSTensor)
#     return MPSTensor(A.A*n)
# end

# function Base.:*(n::Number, A::MPSTensor)
#     return MPSTensor(A.A*n)
# end

# function Base.:*(n::Number, A::CompositeMPSTensor)
#     return CompositeMPSTensor(n*A.A)
# end

# function Base.:/(A::AbstractMPOTensor, α::Number)
#     return  (1/α) * A
# end

# function Base.:/(A::CompositeMPSTensor, n::Number)
#     return (1/n) * A
# end

# function Base.:/(A::MPSTensor, n::Number)
#     @assert n ≠ 0
#     return (1/n)*A
# end


# function normalize!(obj::Union{DenseMPOTensor,MPSTensor})
#     tmp = norm(obj.A)
#     obj.A = obj.A / tmp
#     return tmp
# end

#= function Base.:*(A::CompositeMPSTensor{2, 4}, B::AdjointCompositeMPSTensor{2, 4})
    return @tensor A.A[1,2,3,4] * B.A[4,1,2,3]
end

function Base.:*(A::MPSTensor{3}, Ad::AdjointMPSTensor{3})
    return @tensor A.A[1,2,3] * Ad.A[3,1,2]
end

function Base.:*(A::CompositeMPOTensor{2,6}, B::AdjointCompositeMPOTensor{2,6})
    return  @tensor A.A[1,2,3,4,5,6] * B.A[4,5,6,1,2,3]
end

function Base.:*(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4})
    return  @tensor A.A[1,2,3,4] * B.A[3,4,1,2]
end =#
