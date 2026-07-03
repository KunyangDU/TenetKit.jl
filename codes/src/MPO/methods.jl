# function _funcDenseMPO(func::Function, PhySpaces::AbstractVector, AuxSpaces::AbstractVector; isdisk::Bool=IS_DISK[])
#     length(PhySpaces) == length(AuxSpaces) && push!(AuxSpaces, trivial(PhySpaces[1]))
#     @assert length(PhySpaces) + 1 == length(AuxSpaces)
#     tmp = [DenseMPOTensor(func(PhySpaces[i] ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpaces[i])) for i in eachindex(PhySpaces)]
#     return DenseMPO(tmp; isdisk=isdisk)
# end

# function _funcDenseMPO(func::Function, PhySpace::ElementarySpace, AuxSpaces::AbstractVector; kwargs...)
#     return _funcDenseMPO(func, repeat([PhySpace,],length(AuxSpaces)), AuxSpaces; kwargs...)
# end

function IdDenseMPO(PhySpace::ElementarySpace, AuxSpaces::AbstractVector; isdisk::Bool=IS_DISK[])
    tmp = [DenseMPOTensor(isometry(PhySpace ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpace)) for i in eachindex(AuxSpaces)[1:end-1]]
    return DenseMPO(tmp; isdisk=isdisk)
end

function RandnDenseMPO(PhySpace::ElementarySpace, AuxSpaces::AbstractVector; isdisk::Bool=IS_DISK[])
    tmp = [DenseMPOTensor(randn(PhySpace ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpace)) for i in eachindex(AuxSpaces)[1:end-1]]
    return DenseMPO(tmp; isdisk=isdisk)
end

# function IdDenseMPO(L::Int64, PhySpace::ElementarySpace = ℂ^1, AuxSpace::ElementarySpace = (ℂ^1)'; kwargs...)
#     return _funcDenseMPO(ones, map(x -> repeat([x,],L),(PhySpace,AuxSpace))...; kwargs...)
# end

# function RandDenseMPO(L::Int64, PhySpace::ElementarySpace = ℂ^1, AuxSpace::ElementarySpace = (ℂ^1)'; kwargs...)
#     return _funcDenseMPO(randn, map(x -> repeat([x,],L),(PhySpace,AuxSpace))...; kwargs...)
# end



