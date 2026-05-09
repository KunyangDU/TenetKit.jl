


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
    ts::Vector{DenseMPOTensor}
    center::Vector{Int64}
    
    function DenseMPO(A::Vector{DenseMPOTensor},center::Vector{Int64})
        return new{length(A)}(A,center)
    end

    function DenseMPO(A::Vector{DenseMPOTensor{R}}) where R
        return new{length(A)}(A,[1,length(A)])
    end

    function DenseMPO(t::DenseMPOTensor)
        A = convert(Vector{DenseMPOTensor},[t])
        return new{1}(A,[1,1])        
    end

    function DenseMPO(t::Vector)
        tmp = map(DenseMPOTensor,t)
        A = convert(Vector{DenseMPOTensor},tmp)
        return new{length(A)}(A,[1,length(A)])        
    end
end
const DenseMPQ = Union{DenseMPO,DenseMPS}

mutable struct AdjointMPO{L} <: AbstractMPO
    ts::Vector{AdjointMPOTensor}
    center::Vector{Int64}
    
    function AdjointMPO(A::Vector{AdjointMPOTensor},center::Vector{Int64})
        return new{length(A)}(A,center)
    end

    function AdjointMPO(A::Vector{AdjointMPOTensor{R}}) where R
        return new{length(A)}(A,[1,length(A)])
    end

    function AdjointMPO(t::AdjointMPOTensor)
        A = convert(Vector{AdjointMPOTensor},[t])
        return new{1}(A,[1,1])        
    end

    function AdjointMPO(t::Vector{AbstractTensorMap})
        tmp = map(AdjointMPOTensor,t)
        A = convert(Vector{AdjointMPOTensor},tmp)
        return new{length(A)}(A,[1,length(A)])        
    end
end

Base.adjoint(A::DenseMPO{L}) where {L} = AdjointMPO(deepcopy(adjoint(A.ts)), deepcopy(A.center))
Base.adjoint(A::AdjointMPO{L}) where {L} = DenseMPO(deepcopy(adjoint(A.ts)), deepcopy(A.center))

isadjoint(::DenseMPO) = false
isadjoint(::AdjointMPO) = true

mutable struct RefMPO{L} <: AbstractMPO
    ts::Vector{DenseMPOTensor}
    center::Vector{Int64}
    mapping::Function
    pointer::DenseMPO
    RefMPO(A::DenseMPO{L},mapping::Function = identity) where L = new{L}(A.ts,A.center,mapping,A)
end

issparse(::RefMPO) = false
isadjoint(::RefMPO) = false
isref(::RefMPO) = true
