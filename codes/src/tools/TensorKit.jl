tsvd(A::AbstractTensorMap,I₁::NTuple{N₁,Int64},I₂::NTuple{N₂,Int64};trunc::NamedTuple = (;)) where {N₁,N₂} = svd_trunc(permute(A,(I₁,I₂));trunc = trunc)
leftorth(A::AbstractTensorMap,I₁::NTuple{N₁,Int64},I₂::NTuple{N₂,Int64}) where {N₁,N₂} = left_orth(permute(A,(I₁,I₂)))
rightorth(A::AbstractTensorMap,I₁::NTuple{N₁,Int64},I₂::NTuple{N₂,Int64}) where {N₁,N₂} = right_orth(permute(A,(I₁,I₂)))
# TensorKit.permute(t::T,I₁::NTuple{N₁,Int64},I₂::NTuple{N₂,Int64}) where {N₁,N₂,T <: AbstractTensorMap} = permute(t,(I₁,I₂))
leftorth(A::AbstractTensorMap) = left_orth(A)
rightorth(A::AbstractTensorMap) = right_orth(A)

function TensorKit.dim(A::AbstractTensorMap{<:Union{Float64, ComplexF64}, F}, i::Int64) where F<:GradedSpace
     rcod = numout(A)
     if i ≤ rcod[1]
          D = mapreduce(i -> codomain(A).spaces[i].dims[i], +, eachindex(codomain(A).spaces[i].dims))
          DD = dim(codomain(A).spaces[i])
     else
          D = mapreduce(i -> domain(A).spaces[i - rcod].dims[i], +, eachindex(domain(A).spaces[i - rcod].dims))
          DD = dim(domain(A).spaces[i - rcod])
     end
    return D, DD
end

TensorKit.dim(A::AbstractTensorMap{T,F}, i::Int64) where {T <: Union{Float64, ComplexF64}, F <: Union{CartesianSpace,ComplexSpace}} = dim(space(A, i)), dim(space(A, i))
