



mutable struct DenseMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::Vector{MPSTensor}
    center::Vector{Int64}

    function DenseMPS{L,T}(ts::Vector{MPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(ts,ct)
    end

    function DenseMPS{L,T}(ts::Vector{MPSTensor}) where {L,T}
        return new{L,T}(ts,[1,L])
    end

    function DenseMPS{L,T}(ts::Vector{AbstractTensorMap}) where {L,T}
        return new{L,T}([MPSTensor(t) for t in ts],[1,L])
    end

end

mutable struct AdjointMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::Vector{AdjointMPSTensor}
    center::Vector{Int64}

    function AdjointMPS{L,T}(Elms::Vector{AdjointMPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(Elms,ct)
    end

    function AdjointMPS{L,T}(Elms::Vector{AdjointMPSTensor}) where {L,T}
        return new{L,T}(Elms,[1,length(Elms)])
    end

    function AdjointMPS{L,T}(Elms::Vector{AbstractTensorMap}) where {L,T}
        return new{L,T}([AdjointMPSTensor(elm) for elm in Elms],[1,length(Elms)])
    end

end

function Base.length(::DenseMPS{L,T}) where {L,T}
    return L
end

function Base.adjoint(A::DenseMPS{L,T}) where {L,T}
    return AdjointMPS{L,T}(deepcopy(adjoint(A.ts)), deepcopy(A.center))
end

"""
Generate rand DenseMPS for initial state.
"""
function randMPS(PhySpaces::Vector,AuxSpaces::Vector;
    type::Type = Float64,tailSpace::ElementarySpace = trivial(PhySpaces[1]))
    @assert (L = length(PhySpaces)) == length(AuxSpaces)
    push!(AuxSpaces, tailSpace)
    tmp = Vector{MPSTensor}(undef,L)
    for i in 1:L
        tmp[i] = MPSTensor(randn,AuxSpaces[i] ⊗ PhySpaces[i],AuxSpaces[i+1])
    end

    obj = DenseMPS{L,type}(tmp)

    canonicalize!(obj, L)
    canonicalize!(obj, 1)
    normalize!(obj)

    return obj
end


function randMPS(PhySpace::IndexSpace,AuxSpaces::Vector;kwargs...)
    return randMPS([PhySpace for i in eachindex(AuxSpaces)],AuxSpaces;kwargs...)
end

function AdjointMPSTensor(func, A::MPSTensor{3})
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    tmp = AdjointMPSTensor(func,dm,cdm)
    normalize!(tmp)
    return tmp
end

function AdjointMPSTensor(func,A::MPSTensor{3}, λ::Number,direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if direction == :left
        tmp = AdjointMPSTensor(func,ℂ^(round(Int64,λ * dims(dm)[1])),cdm)
    elseif direction == :right
        tmp = AdjointMPSTensor(func,dm,ℂ^(round(Int64,λ * dims(cdm)[1])) ⊗ cdm[2])
    end
    normalize!(tmp)
    return tmp
end


function getPhySpace(t::DenseMPS)
    return getPhySpace(t.ts[1])
end

function getPhySpace(t::MPSTensor{R}) where R
    if 3 ≤ R
        return codomain(t.A)[2]
    else
        return nothing
    end
end

function getAuxSpace(t::DenseMPS)
    return getAuxSpace(t.ts[1])
end

function getAuxSpace(t::MPSTensor)
    return collect(codomain(t.A))[1], collect(domain(t.A))[end]
end

function getAuxSpace(t::AdjointMPSTensor)
    return collect(domain(t.A))[1], collect(codomain(t.A))[end]
end

function trivial(::GradedSpace{I, D}) where {I, D}
    dims = TensorKit.SortedVectorDict(one(I) => 1)
    return GradedSpace{I,D}(dims, false)
end

function trivial(::ComplexSpace)
    return ℂ^1
end


