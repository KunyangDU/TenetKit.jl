
mutable struct SparseMPO{L}
    ts::Vector{SparseMPOTensor}
    D::NTuple{L,NTuple{3,Int64}}

    function SparseMPO(ts::Vector{<:SparseMPOTensor})
        D_tuple = ntuple(i -> (Int64(length(ts[i].left.fwd)), Int64(length(ts[i].A)), Int64(length(ts[i].right.rev))), length(ts))
        return new{length(ts)}(ts, D_tuple)
    end

    function SparseMPO(t::SparseMPOTensor{DL,D,DR}) where {DL,D,DR}
        return new{1}([t], ((Int64(DL), Int64(D), Int64(DR)),))
    end

    function SparseMPO(ts::Vector{<:SparseMPOTensor}, D::NTuple{L,NTuple{3,Int64}}) where L
        return new{L}(ts, D)
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
    RefMPO(A::DenseMPO{L},mapping::Function = identity) where L = new{L}(A.ts,deepcopy(A.center),mapping,A)
end

issparse(::RefMPO) = false
isadjoint(::RefMPO) = false
isref(::RefMPO) = true
cleanup!(::RefMPO) = nothing

# Fallback for non-RefMPO types
isref(::AbstractMPO) = false

function cleanup!(obj::T) where T <: Union{DenseMPO,AdjointMPO}
    if obj.isdisk && obj.ts isa SerializedElementArrays.SerializedElementArray
        dir = SerializedElementArrays.pathname(obj.ts)
        ispath(dir) && rm(dir; recursive=true, force=true)
    end
    return obj
end
