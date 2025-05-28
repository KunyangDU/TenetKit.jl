



mutable struct DenseMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::Vector{MPSTensor}
    center::Vector{Int64}

    function DenseMPS{L,T}(ts::Vector{MPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(ts,ct)
    end

    function DenseMPS{L,t}(ts::Vector{T}) where T <: Union{MPSTensor,MPSTensor{R}} where {L,t,R}
        return new{L,t}(ts,[1,L])
    end

    function DenseMPS(ts::Vector{T}) where T <: Union{MPSTensor,MPSTensor{R}} where R
        L = length(ts)
        t = eltype(ts[1].A)
        return new{L,t}(ts,[1,L])
    end

    function DenseMPS{L,T}(ts::Vector{AbstractTensorMap}) where {L,T}
        return new{L,T}([MPSTensor(t) for t in ts],[1,L])
    end

end

mutable struct AdjointMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::Vector{AdjointMPSTensor}
    center::Vector{Int64}

    function AdjointMPS{L,T}(ts::Vector{AdjointMPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(ts,ct)
    end

    function AdjointMPS{L,T}(ts::Vector{AdjointMPSTensor}) where {L,T}
        return new{L,T}(ts,[1,length(ts)])
    end

    function AdjointMPS{L,T}(ts::Vector{AbstractTensorMap}) where {L,T}
        return new{L,T}([AdjointMPSTensor(elm) for elm in ts],[1,length(ts)])
    end

end

Base.adjoint(A::DenseMPS{L,T}) where {L,T} = AdjointMPS{L,T}(deepcopy(adjoint(A.ts)), deepcopy(A.center))
Base.adjoint(A::AdjointMPS{L,T}) where {L,T} = DenseMPS{L,T}(deepcopy(adjoint(A.ts)), deepcopy(A.center))







