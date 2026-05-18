

mutable struct SparseMPO{L} <: AbstractMPO
    ts::Vector{SparseMPOTensor}
    D::Vector{NTuple{2,Int64}}

    function SparseMPO(ts::Vector{SparseMPOTensor},
        D::Vector{NTuple{2,Int64}})
        return new{length(ts)}(ts,D)
    end

    function SparseMPO(ts::Vector{T}) where T <: Union{SparseMPOTensor,SparseMPOTensor{N,M}} where {N,M}
        D = map(size,ts)
        return new{length(ts)}(ts,convert(Vector{NTuple{2,Int64}},D))
    end

    function SparseMPO(t::SparseMPOTensor{N,M}) where {N,M}
        D = convert(Vector{NTuple{2,Int64}},[(N,M)])
        ts = convert(Vector{SparseMPOTensor},[t])
        return new{length(ts)}(ts,D)
    end
end

mutable struct DenseMPO{L} <: AbstractMPO
    ts::AbstractVector{DenseMPOTensor}
    center::Vector{Int64}
    isdisk::Bool

    function DenseMPO(A::Vector{DenseMPOTensor},center::Vector{Int64}; isdisk::Bool=IS_DISK[])
        if isdisk
            A = _disk(A)
        end
        return new{length(A)}(A,center,isdisk)
    end

    function DenseMPO(A::Vector{DenseMPOTensor{R}}; isdisk::Bool=IS_DISK[]) where R
        if isdisk
            A = _disk(A)
        end
        return new{length(A)}(A,[1,length(A)],isdisk)
    end

    function DenseMPO(t::DenseMPOTensor; isdisk::Bool=IS_DISK[])
        A = convert(Vector{DenseMPOTensor},[t])
        if isdisk
            A = _disk(A)
        end
        return new{1}(A,[1,1],isdisk)
    end

    function DenseMPO(t::Vector; isdisk::Bool=IS_DISK[])
        tmp = map(DenseMPOTensor,t)
        A = convert(Vector{DenseMPOTensor},tmp)
        if isdisk
            A = _disk(A)
        end
        return new{length(A)}(A,[1,length(A)],isdisk)
    end
end
const DenseMPQ = Union{DenseMPO,DenseMPS}

mutable struct AdjointMPO{L} <: AbstractMPO
    ts::AbstractVector{AdjointMPOTensor}
    center::Vector{Int64}
    isdisk::Bool

    function AdjointMPO(A::Vector{AdjointMPOTensor},center::Vector{Int64}; isdisk::Bool=IS_DISK[])
        if isdisk
            A = _disk(A)
        end
        return new{length(A)}(A,center,isdisk)
    end

    function AdjointMPO(A::Vector{AdjointMPOTensor{R}}; isdisk::Bool=IS_DISK[]) where R
        if isdisk
            A = _disk(A)
        end
        return new{length(A)}(A,[1,length(A)],isdisk)
    end

    function AdjointMPO(t::AdjointMPOTensor; isdisk::Bool=IS_DISK[])
        A = convert(Vector{AdjointMPOTensor},[t])
        if isdisk
            A = _disk(A)
        end
        return new{1}(A,[1,1],isdisk)
    end

    function AdjointMPO(t::Vector{AbstractTensorMap}; isdisk::Bool=IS_DISK[])
        tmp = map(AdjointMPOTensor,t)
        A = convert(Vector{AdjointMPOTensor},tmp)
        if isdisk
            A = _disk(A)
        end
        return new{length(A)}(A,[1,length(A)],isdisk)
    end
end

Base.adjoint(A::DenseMPO{L}) where {L} = AdjointMPO(adjoint(A.ts), deepcopy(A.center); isdisk=A.isdisk)
Base.adjoint(A::AdjointMPO{L}) where {L} = DenseMPO(adjoint(A.ts), deepcopy(A.center); isdisk=A.isdisk)

isadjoint(::DenseMPO) = false
isadjoint(::AdjointMPO) = true

mutable struct RefMPO{L} <: AbstractMPO
    ts::AbstractVector{DenseMPOTensor}
    center::Vector{Int64}
    mapping::Function
    pointer::DenseMPO
    RefMPO(A::DenseMPO{L},mapping::Function = identity) where L = new{L}(A.ts,A.center,mapping,A)
end

issparse(::RefMPO) = false
isadjoint(::RefMPO) = false
isref(::RefMPO) = true

# Fallback for non-RefMPO types
isref(::AbstractMPO) = false

function cleanup!(obj::T) where T <: Union{DenseMPO,AdjointMPO}
    if obj.isdisk && obj.ts isa SerializedElementArrays.SerializedElementArray
        dir = SerializedElementArrays.pathname(obj.ts)
        ispath(dir) && rm(dir; recursive=true, force=true)
    end
    return obj
end
