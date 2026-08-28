
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
    TensorKit.axpy!(α, A.A, B.A)   # 零分配：要求 B.A 类型够宽（需提升用 add!!）
    return B
end

function axpby!(α::Number, A::AbstractTensorWrapper, β::Number, B::AbstractTensorWrapper)
    TensorKit.axpby!(α, A.A, β, B.A)   # 零分配：要求 B.A 类型够宽（需提升用 add!!）
    return B
end
axpby!(α::Number, A::AbstractTensorWrapper, ::Number, ::Nothing) = α * A
axpby!(::Number, ::Nothing, β::Number, A::AbstractTensorWrapper) = rmul!(A, β)
axpy!(α::Number, A::AbstractTensorWrapper, ::Nothing) = α * A
axpy!(::Number, ::Nothing, B::AbstractTensorWrapper) = B

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
TensorKit.space(A::AbstractTensorWrapper,i::Int64) = space(A.A,i)
TensorKit.space(A::AbstractLocalOperator,i::Int64) = space(A.A,i)

TensorKit.dims(A::AbstractTensorWrapper) = dims(A.A)

issparse(::T) where T <: Union{DenseMPS,AdjointMPS,DenseMPO,AdjointMPO} = false
issparse(::SparseMPO) = true
issparse(::SparseMPOTensor) = true

_isdisk(obj::T) where T <: Union{DenseMPS,AdjointMPS,DenseMPO,AdjointMPO} = obj.isdisk
_isdisk(::SparseMPO) = false
_isdisk(::RefMPO) = false
_isdisk(::RefMPS) = false
Base.size(t::DenseMPOTensor{4}) = map(dim,t.A |> x -> (codomain(x)[2],domain(x)[1]))
Base.length(::DenseMPO{L}) where L = L
Base.length(::AdjointMPO{L}) where L = L
Base.length(::SparseMPO{L}) where L = L
Base.length(::DenseMPS{L}) where L = L
Base.length(::AdjointMPS{L}) where L = L
Base.length(::RefMPS{L}) where L = L
Base.length(::RefMPO{L}) where L = L

Base.firstindex(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = 1
Base.lastindex(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = lastindex(obj.ts)
Base.size(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = (lastindex(obj),)
Base.axes(obj::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = Base.OneTo(lastindex(obj))
Base.firstindex(::RefMPS) = 1
Base.lastindex(obj::RefMPS) = lastindex(obj.ts)
Base.size(obj::RefMPS) = (lastindex(obj),)
Base.axes(obj::RefMPS) = Base.OneTo(lastindex(obj))

Base.firstindex(obj::RefMPO) = 1
Base.lastindex(obj::RefMPO) = lastindex(obj.ts)
Base.size(obj::RefMPO) = (lastindex(obj),)
Base.axes(obj::RefMPO) = Base.OneTo(lastindex(obj))
Base.size(::SparseMPOTensor{DL,D,DR,T}) where {DL,D,DR,T} = DL,DR

function normalize!(obj::Union{DenseMPO{L},DenseMPS{L},AdjointMPO{L},AdjointMPS{L}}) where L
    @assert (site = obj.center[1]) == obj.center[2]
    t = obj[site]         # 磁盘对象: 反序列化; 内存对象: 引用
    tmp = normalize!(t)   # 就地归一化
    obj[site] = t         # 磁盘对象: 序列化写回; 内存对象: 无操作
    return tmp
end

function normalize!(obj::RefMPS)
    @assert (site = obj.center[1]) == obj.center[2]
    return normalize!(obj[site])  # RefMPS 的 setindex! 是空操作, 无需写回
end

function normalize!(obj::RefMPO)
    @assert (site = obj.center[1]) == obj.center[2]
    return normalize!(obj[site])
end

function TensorKit.norm(obj::Union{DenseMPO{L},DenseMPS{L},AdjointMPO{L},AdjointMPS{L}}) where L
    @assert (site = obj.center[1]) == obj.center[2]
    return norm(obj[site])
end

function TensorKit.norm(obj::RefMPS)
    @assert (site = obj.center[1]) == obj.center[2]
    return norm(obj[site])
end

function TensorKit.norm(obj::RefMPO)
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
Base.:*(B::T,A::Number) where T <: AbstractTensorWrapper = T(A * B.A)
Base.:/(A::T,B::Number) where T <: AbstractTensorWrapper = (1/B) * A

scale(t::Tuple{T₁,T₂}) where {T₁ <: AbstractTensorWrapper,T₂ <: Number} = scale((t[1].A,t[2]))
scale!!(t::Tuple{T₁,T₂}) where {T₁ <: AbstractTensorWrapper,T₂ <: Number} = scale!!((t[1].A,t[2]))
scale!!(t::Tuple{T₁,T₂,T₃}) where {T₁ <: AbstractTensorWrapper,T₂ <: AbstractTensorWrapper,T₃ <: Number} = scale!!((t[1].A,t[2].A,t[3]))
TensorKit.zerovector(t::Tuple{T₁,T₂}) where {T₁ <: AbstractTensorWrapper,T₂ <: Number} = zerovector((t[1].A,t[2]))
add!!(t::Tuple{T₁,T₂,T₃}) where {T₁ <: AbstractTensorWrapper,T₂ <: AbstractTensorWrapper,T₃ <: Number} = add!!((t[1].A,t[2].A,t[3],t[4]))

TensorKit.codomain(A::AbstractTensorWrapper) = codomain(A.A)
TensorKit.domain(A::AbstractTensorWrapper) = domain(A.A)
Base.eltype(A::AbstractTensorWrapper) = eltype(A.A)
Base.randn(A::T) where T <: AbstractTensorWrapper = T(TensorMap(randn, eltype(A), codomain(A), domain(A)))

Base.copy(A::T) where T <: AbstractTensorWrapper = T(copy(A.A))

TensorKit.numind(A::T) where T <: AbstractTensorWrapper = numind(A.A)

Base.getindex(obj::T, i::Int64) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "deserialize" obj.ts[i]) : obj.ts[i]
Base.getindex(obj::T, i::Int64) where T <: Union{RefMPO, RefMPS} = obj.mapping(obj.ts[i])
Base.getindex(obj::T, stp::UnitRange) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "deserialize" [obj.ts[i] for i in stp]) : [obj.ts[i] for i in stp]
Base.getindex(obj::T, stp::UnitRange) where T <: Union{RefMPO, RefMPS} = obj.mapping.([obj.ts[i] for i in stp])
Base.setindex!(obj::T, val, i::Int64) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "serialize" obj.ts[i] = val) : (obj.ts[i] = val)
Base.setindex!(obj::T, vals, stp::UnitRange) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (@timeit _local_io_timer() "serialize" for (i, v) in zip(stp, vals); obj.ts[i] = v; end) : (for (i, v) in zip(stp, vals); obj.ts[i] = v; end)
Base.setindex!(::RefMPO, val, i::Int64) = nothing
Base.setindex!(::RefMPS, val, i::Int64) = nothing
Base.setindex!(::RefMPS, vals, stp::UnitRange) = nothing

Base.getindex(obj::T, ::Colon) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? [obj.ts[i] for i in 1:length(obj.ts)] : obj.ts[:]
Base.getindex(obj::T, ::Colon) where T <: Union{RefMPO, RefMPS} = obj.mapping.(obj.ts[:])
Base.setindex!(obj::T, vals, ::Colon) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS,SparseMPO} = _isdisk(obj) ? (for (i, v) in enumerate(vals); obj.ts[i] = v; end) : (obj.ts[:] = vals)
Base.setindex!(::RefMPS, vals, ::Colon) = nothing

Base.getindex(obj::T, i::Int64) where T <: Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor} = obj.A[i]
Base.getindex(obj::T, inds::AbstractVector{Int64}) where T <: Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor} = [obj.A[i] for i in inds]
Base.getindex(obj::T, i::Int...) where T <: Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor} = obj.A[i...]

Base.setindex!(obj::SparseLeftEnvironmentTensor, A::LeftEnvironmentTensor,i::Int64) = (obj.A[i] = A)
Base.setindex!(obj::SparseRightEnvironmentTensor, A::RightEnvironmentTensor,i::Int64) = (obj.A[i] = A)
Base.setindex!(obj::T, val, i::Int...) where T <: Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor} = (obj.A[i...] = val)

Base.length(obj::T) where T <: Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor} = length(obj.A)
Base.size(obj::T) where T <: Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor} = size(obj.A)

Base.iterate(obj::T, args...) where T <: Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor} = iterate(obj.A, args...)



function Base.:+(A::LeftEnvironmentTensor,
    B::LeftEnvironmentTensor)
    return LeftEnvironmentTensor(A.A + B.A)
end

function Base.:+(A::RightEnvironmentTensor,
    B::RightEnvironmentTensor)
    return RightEnvironmentTensor(A.A + B.A)
end

Base.:*(α::Number, A::LeftEnvironmentTensor) = LeftEnvironmentTensor(α * A.A)
Base.:*(α::Number, A::RightEnvironmentTensor) = RightEnvironmentTensor(α * A.A)

axpy!(α::Number, A::LeftEnvironmentTensor, B::LeftEnvironmentTensor) = (TensorKit.axpy!(α, A.A, B.A); B)
axpy!(α::Number, A::RightEnvironmentTensor, B::RightEnvironmentTensor) = (TensorKit.axpy!(α, A.A, B.A); B)
axpy!(α::Number, A::LeftEnvironmentTensor, ::Nothing) = α * A
axpy!(α::Number, A::RightEnvironmentTensor, ::Nothing) = α * A
axpy!(::Number, ::Nothing, B::LeftEnvironmentTensor) = B
axpy!(::Number, ::Nothing, B::RightEnvironmentTensor) = B

Base.length(::Union{DenseMPS{L}, DenseMPO{L}, RefMPS{L}, RefMPO{L}}) where L = L
