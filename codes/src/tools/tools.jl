_getD(A::DenseMPOTensor{4}) = dims(domain(A.A))[1]
_getD(A::LeftEnvironmentTensor) = dims(domain(A.A))[end]
_getD(A::RightEnvironmentTensor) = dims(codomain(A.A))[end]
_getD(A::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor}) = _getD(A.A[1])
_getD(A::Union{CompositeMPSTensor,CompositeMPOTensor}) = dims(domain(A.A))[1]

function Main.dims(A::Union{LeftEnvironmentTensor,RightEnvironmentTensor,
    DenseMPOTensor,MPSTensor,
    LeftCompositeEnvironmentTensor,RightCompositeEnvironmentTensor})
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

_maxdim(obj::Union{DenseMPS,DenseMPO}) = _maxdim(obj.ts)
_maxdim(obj::AbstractVector{MPSTensor}) = maximum(map(x -> dims(x) |> y -> max(y[1][1],y[2][1]),obj))
_maxdim(obj::AbstractVector{DenseMPOTensor}) = maximum(map(x -> dims(x) |> y -> max(y[1][2],y[2][1]),obj))

Base.:≈(A::Tuple,B::Tuple) = collect(A) ≈ collect(B) 

getAuxSpace(t::DenseMPOTensor{4}) = collect(codomain(t.A))[2], collect(domain(t.A))[1]
getAuxSpace(t::AdjointMPOTensor{4}) = collect(domain(t.A))[2], collect(codomain(t.A))[1]

Base.length(::DenseMPOTensor) = 1
rank(A::AbstractTensorMap) = length(codomain(A)) + length(domain(A))

composite(A::MPSTensor{3}, B::MPSTensor{3}) = CompositeMPSTensor(@tensor tmp[-1 -2 -3; -4] ≔ A.A[-1,-2,1]*B.A[1,-3,-4])
composite(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}) = CompositeMPOTensor(@tensor tmp[-1 -2 -3;-4 -5 -6] ≔ A.A[-2,-3,1,-6] * B.A[-1,1,-4,-5])
composite(A::T, B::T) where T <: Union{AdjointMPSTensor{3},AdjointMPOTensor{4}} = composite(A',B')'

getPhySpace(t::DenseMPS) = getPhySpace(t[1])
getPhySpace(t::MPSTensor{R}) where R = 3 ≤ R ? codomain(t.A)[2] : nothing
getAuxSpace(t::DenseMPS) = getAuxSpace(t[1])
getAuxSpace(t::MPSTensor) = collect(codomain(t.A))[1], collect(domain(t.A))[end]
getAuxSpace(t::AdjointMPSTensor) = collect(domain(t.A))[1], collect(codomain(t.A))[end]
trivial(::GradedSpace{I, D}) where {I, D} = GradedSpace{I,D}(TensorKit.SortedVectorDict(one(I) => 1), false)
trivial(::ComplexSpace) = ℂ^1
Base.length(::DenseMPS{L,T}) where {L,T} = L

function _isometry(sps::GradedSpace...;T::Type = ComplexF64)
    sp = reduce(⊗, sps)
    tmp = TensorMap(zeros,T,sp,sp)
    return rightorth(tmp)[2]
end

function countmap(obj::Vector)
    counts = Dict{eltype(obj), Int}()
    for element in obj
        counts[element] = get(counts, element, 0) + 1
    end
    return counts
end

isfermionic(A::NTuple{N, Bool}) where N = ((-1)^sum(A) == -1)
# isfermionic(A::Vector{Bool})= ((-1)^sum(A) == -1)

# trivialspace(A::ObservableOperator) = trivialspace(A.A)
# trivialspace(A::LocalOperator) = trivial(A.A)
# trivialspace(A::AbstractTensorMap) = trivial(space(A)[1])

_left_isometry(obj::DenseMPS) = space(obj[1])[1] |> x -> isometry(x,x)
_right_isometry(obj::DenseMPS) = space(obj[end])[3]' |> x -> isometry(x,x)
_left_isometry(obj::DenseMPO) = space(obj[1])[2] |> x -> isometry(x,x)
_right_isometry(obj::DenseMPO) = space(obj[end])[3]' |> x -> isometry(x,x)
