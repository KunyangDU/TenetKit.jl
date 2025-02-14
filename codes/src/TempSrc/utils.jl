_getD(A::DenseMPOTensor{4}) = dims(domain(A.A))[1]
_getD(A::LeftEnvironmentTensor) = dims(domain(A.A))[end]
_getD(A::RightEnvironmentTensor) = dims(codomain(A.A))[end]
_getD(A::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor}) = _getD(A.A[1])
_getD(A::Union{CompositeMPSTensor,CompositeMPOTensor}) = dims(domain(A.A))[1]

function Main.dims(A::Union{LeftEnvironmentTensor,RightEnvironmentTensor,DenseMPOTensor,LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor})
    return dims(codomain(A.A)),dims(domain(A.A))
end

function anisometry(codom,dom)
    physpace = codom[1],dom[2]
    auxspace = codom[2],dom[1]
    atmp = TensorMap(ones,auxspace...)
    ptmp = isometry(physpace...)
    @tensor tmp[-1,-2;-3,-4] ≔ ptmp[-1,-4] * atmp[-2,-3]
    return tmp
end

function easyinterp10(v,N=100)
    return 10. .^ (range(log10.(extrema(v))..., N))
end

function diag(A::AbstractMatrix)
    return [A[i,i] for i in 1:min(size(A)...)]
end