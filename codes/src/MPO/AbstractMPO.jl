



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

mutable struct DenseMPO{L, Sc<:Union{Float64, ComplexF64}, MT<:AbstractMPOTensor} <: AbstractMPO
    ts::Vector{MT}
    center::Vector{Int64}

    function DenseMPO(A::Vector{MT}, center::Vector{Int64}) where {MT<:AbstractMPOTensor}
        Sc = scalartype(A[1].A)
        return new{length(A), Sc, MT}(A, center)
    end

    function DenseMPO(A::Vector{MT}) where {MT<:AbstractMPOTensor}
        Sc = scalartype(A[1].A)
        return new{length(A), Sc, MT}(A, [1,length(A)])
    end

    function DenseMPO(t::MT) where {MT<:AbstractMPOTensor}
        Sc = scalartype(t.A)
        A  = [t]
        return new{1, Sc, MT}(A, [1,1])
    end

    function DenseMPO(t::Vector)
        tmp = map(DenseMPOTensor, t)
        MT  = eltype(tmp)
        Sc  = scalartype(tmp[1].A)
        return new{length(tmp), Sc, MT}(tmp, [1,length(tmp)])
    end
end
const DenseMPQ = Union{DenseMPO,DenseMPS}

mutable struct AdjointMPO{L, Sc<:Union{Float64, ComplexF64}, MT<:AbstractMPOTensor} <: AbstractMPO
    ts::Vector{MT}
    center::Vector{Int64}

    function AdjointMPO(A::Vector{MT}, center::Vector{Int64}) where {MT<:AbstractMPOTensor}
        Sc = scalartype(A[1].A)
        return new{length(A), Sc, MT}(A, center)
    end

    function AdjointMPO(A::Vector{MT}) where {MT<:AbstractMPOTensor}
        Sc = scalartype(A[1].A)
        return new{length(A), Sc, MT}(A, [1,length(A)])
    end

    function AdjointMPO(t::MT) where {MT<:AbstractMPOTensor}
        Sc = scalartype(t.A)
        A  = [t]
        return new{1, Sc, MT}(A, [1,1])
    end

    function AdjointMPO(t::Vector{AbstractTensorMap})
        tmp = map(AdjointMPOTensor, t)
        MT  = eltype(tmp)
        Sc  = scalartype(tmp[1].A)
        return new{length(tmp), Sc, MT}(tmp, [1,length(tmp)])
    end
end

Base.adjoint(A::DenseMPO{L,Sc,MT}) where {L,Sc,MT} = AdjointMPO(deepcopy(adjoint(A.ts)), deepcopy(A.center))
Base.adjoint(A::AdjointMPO{L,Sc,MT}) where {L,Sc,MT} = DenseMPO(deepcopy(adjoint(A.ts)), deepcopy(A.center))


