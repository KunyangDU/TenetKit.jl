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

function vonNeumann(S::AbstractTensorMap{<:ElementarySpace,1,1})
    _tmptrace(x) = @tensor x[1,1]
    d = _tmptrace(S*S')
    @assert d != 0
    A = S/d |> x -> x*x'
    return real(_tmptrace(-A*log(A)))
end

_maxdim(obj::Union{DenseMPS,DenseMPO}) = _maxdim(obj.ts)
_maxdim(obj::Vector{MPSTensor}) = maximum(map(x -> dims(x) |> y -> max(y[1][1],y[2][1]),obj))
_maxdim(obj::Vector{DenseMPOTensor}) = maximum(map(x -> dims(x) |> y -> max(y[1][2],y[2][1]),obj))

_getdim(trunc::TensorKit.MultipleTruncation) = filter(x -> typeof(x) <: TensorKit.TruncationDimension,collect(trunc.truncations))[1].dim
_getcutoff(trunc::TensorKit.MultipleTruncation) = filter(x -> typeof(x) <: TensorKit.TruncationCutoff,collect(trunc.truncations))[1].ϵ
_!getdim(trunc::TensorKit.MultipleTruncation) = filter(x -> typeof(x) != TensorKit.TruncationDimension, collect(trunc.truncations))
_updatedim(trunc::TensorKit.MultipleTruncation,ratio::Number) = TensorKit.MultipleTruncation(tuple(truncdim(ceil(Int64,_getdim(trunc)*ratio)),_!getdim(trunc)...))

_getdim(trunc::TensorKit.TruncationDimension) = trunc.dim
TensorKit.truncdim(tc::Union{TensorKit.MultipleTruncation,TensorKit.TruncationDimension},ratio::Number) = truncdim(ceil(Int64,_getdim(tc)*ratio))
TensorKit.truncdim(trunc::TruncationScheme) = truncdim(_getdim(trunc))

Base.:≈(A::Tuple,B::Tuple) = collect(A) ≈ collect(B) 

getAuxSpace(t::DenseMPOTensor{4}) = collect(codomain(t.A))[2], collect(domain(t.A))[1]
getAuxSpace(t::AdjointMPOTensor{4}) = collect(domain(t.A))[2], collect(codomain(t.A))[1]

Base.length(::DenseMPOTensor) = 1
rank(A::AbstractTensorMap) = length(codomain(A)) + length(domain(A))

composite(A::MPSTensor{3}, B::MPSTensor{3}) = CompositeMPSTensor(@tensor tmp[-1 -2 -3; -4] ≔ A.A[-1,-2,1]*B.A[1,-3,-4])
composite(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}) = CompositeMPOTensor(@tensor tmp[-1 -2 -3;-4 -5 -6] ≔ A.A[-2,-3,1,-6] * B.A[-1,1,-4,-5])
composite(A::T, B::T) where T <: Union{AdjointMPSTensor{3},AdjointMPOTensor{4}} = composite(A',B')'

getPhySpace(t::DenseMPS) = getPhySpace(t.ts[1])
getPhySpace(t::MPSTensor{R}) where R = 3 ≤ R ? codomain(t.A)[2] : nothing
getAuxSpace(t::DenseMPS) = getAuxSpace(t.ts[1])
getAuxSpace(t::MPSTensor) = collect(codomain(t.A))[1], collect(domain(t.A))[end]
getAuxSpace(t::AdjointMPSTensor) = collect(domain(t.A))[1], collect(codomain(t.A))[end]
trivial(::GradedSpace{I, D}) where {I, D} = GradedSpace{I,D}(TensorKit.SortedVectorDict(one(I) => 1), false)
trivial(::ComplexSpace) = ℂ^1
Base.length(::DenseMPS{L,T}) where {L,T} = L

