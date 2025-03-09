function _funcDenseMPO(func::Function, PhySpaces::AbstractVector, AuxSpaces::AbstractVector)
    length(PhySpaces) == length(AuxSpaces) && push!(AuxSpaces, trivial(PhySpaces[1]))
    @assert length(PhySpaces) + 1 == length(AuxSpaces)
    tmp = [DenseMPOTensor(TensorMap(func,PhySpaces[i] ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpaces[i])) for i in eachindex(PhySpaces)]
    return DenseMPO(tmp)
end

function _funcDenseMPO(func::Function, PhySpace::ElementarySpace, AuxSpaces::AbstractVector)
    return _funcDenseMPO(func, repeat([PhySpace,],length(AuxSpaces)), AuxSpaces)
end

function IdDenseMPO(PhySpace::ElementarySpace, AuxSpaces::AbstractVector)
    tmp = [DenseMPOTensor(isometry(PhySpace ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpace)) for i in eachindex(AuxSpaces)[1:end-1]]
    return DenseMPO(tmp)
end

function IdDenseMPO(L::Int64, PhySpace::ElementarySpace = ℂ^1, AuxSpace::ElementarySpace = (ℂ^1)')
    return _funcDenseMPO(ones, map(x -> repeat([x,],L),(PhySpace,AuxSpace))...)
end

function RandDenseMPO(L::Int64, PhySpace::ElementarySpace = ℂ^1, AuxSpace::ElementarySpace = (ℂ^1)')
    return _funcDenseMPO(randn, map(x -> repeat([x,],L),(PhySpace,AuxSpace))...)
end



