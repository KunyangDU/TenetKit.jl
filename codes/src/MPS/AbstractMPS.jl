




mutable struct DenseMPS{L, Sc<:Union{Float64, ComplexF64}, MT<:AbstractMPSTensor} <: AbstractMPS
    ts::Vector{MT}
    center::Vector{Int64}

    function DenseMPS{L,Sc,MT}(ts::Vector{MT},
        ct::Vector{Int64}) where {L, Sc, MT<:AbstractMPSTensor}
        return new{L,Sc,MT}(ts,ct)
    end

    function DenseMPS{L,Sc}(ts::Vector{MT}) where {L, Sc, MT<:AbstractMPSTensor}
        return new{L,Sc,MT}(ts,[1,L])
    end

    function DenseMPS(ts::Vector{MT}) where {MT<:AbstractMPSTensor}
        L  = length(ts)
        Sc = scalartype(ts[1].A)
        return new{L,Sc,MT}(ts,[1,L])
    end

    function DenseMPS{L,Sc}(ts::Vector{AbstractTensorMap}) where {L, Sc}
        mts = [MPSTensor(t) for t in ts]
        MT  = eltype(mts)
        return new{L,Sc,MT}(mts,[1,L])
    end

end

mutable struct AdjointMPS{L, Sc<:Union{Float64, ComplexF64}, MT<:AbstractMPSTensor} <: AbstractMPS
    ts::Vector{MT}
    center::Vector{Int64}

    function AdjointMPS{L,Sc,MT}(ts::Vector{MT},
        ct::Vector{Int64}) where {L, Sc, MT<:AbstractMPSTensor}
        return new{L,Sc,MT}(ts,ct)
    end

    function AdjointMPS{L,Sc,MT}(ts::Vector{MT}) where {L, Sc, MT<:AbstractMPSTensor}
        return new{L,Sc,MT}(ts,[1,length(ts)])
    end

    function AdjointMPS{L,Sc}(ts::Vector{AbstractTensorMap}) where {L, Sc}
        mts = [AdjointMPSTensor(t) for t in ts]
        MT  = eltype(mts)
        return new{L,Sc,MT}(mts,[1,length(mts)])
    end

end

Base.adjoint(A::DenseMPS{L,Sc,MT}) where {L,Sc,MT} = AdjointMPS{L,Sc}(deepcopy(adjoint(A.ts)), deepcopy(A.center))
Base.adjoint(A::AdjointMPS{L,Sc,MT}) where {L,Sc,MT} = DenseMPS(deepcopy(adjoint(A.ts)), deepcopy(A.center))

# Internal constructor for adjoint that takes an already-constructed vector + center
function DenseMPS(ts::Vector{MT}, ct::Vector{Int64}) where {MT<:AbstractMPSTensor}
    L  = length(ts)
    Sc = scalartype(ts[1].A)
    return DenseMPS{L,Sc,MT}(ts, ct)
end
function AdjointMPS{L,Sc}(ts::Vector{MT}, ct::Vector{Int64}) where {L, Sc, MT<:AbstractMPSTensor}
    return AdjointMPS{L,Sc,MT}(ts, ct)
end
