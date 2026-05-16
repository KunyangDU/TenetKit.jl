



mutable struct DenseMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::AbstractVector{MPSTensor}
    center::Vector{Int64}
    isdisk::Bool

    function DenseMPS{L,T}(ts::AbstractVector{MPSTensor},
        ct::Vector{Int64}; isdisk::Bool=false) where {L,T}
        if isdisk && ts isa Vector
            ts = SerializedElementArrays.disk(ts)
        end
        return new{L,T}(ts,ct,isdisk)
    end

    function DenseMPS{L,t}(ts::Vector{T}; isdisk::Bool=false) where T <: Union{MPSTensor,MPSTensor{R}} where {L,t,R}
        if isdisk
            ts = SerializedElementArrays.disk(ts)
        end
        return new{L,t}(ts,[1,L],isdisk)
    end

    function DenseMPS(ts::Vector{T}; isdisk::Bool=false) where T <: Union{MPSTensor,MPSTensor{R}} where R
        L = length(ts)
        t = eltype(ts[1].A)
        if isdisk
            ts = SerializedElementArrays.disk(ts)
        end
        return new{L,t}(ts,[1,L],isdisk)
    end

    function DenseMPS{L,T}(ts::Vector{AbstractTensorMap}; isdisk::Bool=false) where {L,T}
        v = [MPSTensor(t) for t in ts]
        if isdisk
            v = SerializedElementArrays.disk(v)
        end
        return new{L,T}(v,[1,L],isdisk)
    end

end

mutable struct AdjointMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::AbstractVector{AdjointMPSTensor}
    center::Vector{Int64}
    isdisk::Bool

    function AdjointMPS{L,T}(ts::AbstractVector{AdjointMPSTensor},
        ct::Vector{Int64}; isdisk::Bool=false) where {L,T}
        if isdisk && ts isa Vector
            ts = SerializedElementArrays.disk(ts)
        end
        return new{L,T}(ts,ct,isdisk)
    end

    function AdjointMPS{L,T}(ts::Vector{AdjointMPSTensor}; isdisk::Bool=false) where {L,T}
        if isdisk
            ts = SerializedElementArrays.disk(ts)
        end
        return new{L,T}(ts,[1,length(ts)],isdisk)
    end

    function AdjointMPS{L,T}(ts::Vector{AbstractTensorMap}; isdisk::Bool=false) where {L,T}
        v = [AdjointMPSTensor(elm) for elm in ts]
        if isdisk
            v = SerializedElementArrays.disk(v)
        end
        return new{L,T}(v,[1,length(v)],isdisk)
    end

end

Base.adjoint(A::DenseMPS{L,T}) where {L,T} = AdjointMPS{L,T}(deepcopy(adjoint(A.ts)), deepcopy(A.center); isdisk=A.isdisk)
Base.adjoint(A::AdjointMPS{L,T}) where {L,T} = DenseMPS{L,T}(deepcopy(adjoint(A.ts)), deepcopy(A.center); isdisk=A.isdisk)







