
function TensorKit.scalartype(A::AbstractTensorWrapper)
    return TensorKit.scalartype(A.A)
end

Base.similar(A::AbstractTensorWrapper, ::Type{S}) where {S<:Number} = zerovector(A, S)
function zerovector(A::T, ::Type{S}) where {S<:Number, T<:AbstractTensorWrapper}
    return convert(T, TensorKit.zerovector(A.A, S))
end  

Base.convert(::Type{T}, A::AbstractTensorMap) where {T<:AbstractTensorWrapper} = T(A)

function zerovector!(A::AbstractTensorWrapper) 
    TensorKit.zerovector!(A.A)
    return A
end
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
    axpy!(α, A.A, B.A)
    return B
end
axpy!(::Number, ::Nothing, A::AbstractTensorWrapper) = A
axpy!(α::Number, A::AbstractTensorWrapper, ::Nothing) = α * A
axpby!(α::Number, ::Nothing, β::Number, A::AbstractTensorWrapper) = rmul!(A, β)
axpby!(α::Number, A::AbstractTensorWrapper, β::Number, ::Nothing) = axpy!(α, A, nothing)

add!(A::AbstractTensorWrapper, B::AbstractTensorWrapper) = axpy!(true, B, A)
add!(A::AbstractTensorWrapper, ::Nothing) = A
add!(::Nothing, A::AbstractTensorWrapper) = A

scale!(A::AbstractTensorWrapper, α::Number) = rmul!(A, α)
scale(A::AbstractTensorWrapper, α::Number) = α * A

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

Base.:*(A::AbstractTensorWrapper,B::AbstractTensorWrapper) = A.A * B.A
Base.isapprox(A::AbstractTensorWrapper,B::AbstractTensorWrapper) = isapprox(A.A , B.A)
