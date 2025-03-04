

function TensorKit.scalartype(A::AbstractMPSTensor)
    return TensorKit.scalartype(A.A)
end

Base.similar(A::AbstractMPSTensor, ::Type{S}) where {S<:Number} = zerovector(A, S)
function zerovector(A::T, ::Type{S}) where {S<:Number, T<:AbstractMPSTensor}
    return convert(T, TensorKit.zerovector(A.A, S))
end  

Base.convert(::Type{T}, A::AbstractTensorMap) where {T<:AbstractMPSTensor} = T(A)

function zerovector!(A::AbstractMPSTensor) 
    TensorKit.zerovector!(A.A)
    return A
end
function TensorKit.LinearAlgebra.rmul!(A::AbstractMPSTensor, α::Number)
    TensorKit.LinearAlgebra.rmul!(A.A, α)
    return A
end
function TensorKit.LinearAlgebra.mul!(A::T, B::T, α::Number) where {T<:AbstractMPSTensor}
    TensorKit.LinearAlgebra.mul!(A.A, B.A, α)
    return A
end

function add!!(A::AbstractMPSTensor,
    B::AbstractMPSTensor,
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

function axpy!(α::Number, A::T, B::T) where {T<:AbstractMPSTensor}
    axpy!(α, A.A, B.A)
    return B
end
axpy!(::Number, ::Nothing, A::AbstractMPSTensor) = A
axpy!(α::Number, A::AbstractMPSTensor, ::Nothing) = α * A
axpby!(α::Number, ::Nothing, β::Number, A::AbstractMPSTensor) = rmul!(A, β)
axpby!(α::Number, A::AbstractMPSTensor, β::Number, ::Nothing) = axpy!(α, A, nothing)

add!(A::AbstractMPSTensor, B::AbstractMPSTensor) = axpy!(true, B, A)
add!(A::AbstractMPSTensor, ::Nothing) = A
add!(::Nothing, A::AbstractMPSTensor) = A

scale!(A::AbstractMPSTensor, α::Number) = rmul!(A, α)
scale(A::AbstractMPSTensor, α::Number) = α * A

function scale!!(A::AbstractMPSTensor, α::S) where {S<:Number}
    T = promote_type(scalartype(A.A), S)
    if T <: scalartype(A)
         return scale!(A, α)
    else
         return scale(A, α)
    end
end

