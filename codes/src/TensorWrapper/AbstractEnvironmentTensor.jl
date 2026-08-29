
mutable struct RightEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function RightEnvironmentTensor(t::AbstractTensorMap)
        return new{numind(t)}(t)
    end
    function RightEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{numind(t)}(t)
    end
end

mutable struct LeftEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LeftEnvironmentTensor(t::AbstractTensorMap)
        return new{numind(t)}(t)
    end
    function LeftEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{numind(t)}(t)
    end
end

Base.adjoint(A::LeftEnvironmentTensor{2}) = LeftEnvironmentTensor{2}(A.A')
Base.adjoint(A::RightEnvironmentTensor{2}) = RightEnvironmentTensor{2}(A.A')

mutable struct LeftCompositeEnvironmentTensor{Rcd,Rt,N,I} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LeftCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{numout(t),numind(t),3,3}(t)
    end

    function LeftCompositeEnvironmentTensor{n,r}(t::AbstractTensorMap) where {n,r}
        return new{numout(t),numind(t),3,3}(t)
    end

    function LeftCompositeEnvironmentTensor(t::AbstractTensorMap,n::Int64,i::Int64)
        return new{numout(t),numind(t),n,i}(t)
    end

    function LeftCompositeEnvironmentTensor{rcd,rt,n,i}(t::AbstractTensorMap) where {rcd,rt,n,i}
        return new{rcd,rt,n,i}(t)
    end
end

mutable struct RightCompositeEnvironmentTensor{Rcd,Rt,N,I} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function RightCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{numin(t),numind(t),3,3}(t)
    end

    function RightCompositeEnvironmentTensor{n,r}(t::AbstractTensorMap) where {n,r}
        return new{numin(t),numind(t),3,3}(t)
    end
    function RightCompositeEnvironmentTensor(t::AbstractTensorMap,n::Int64,i::Int64)
        return new{numin(t),numind(t),n,i}(t)
    end
    function RightCompositeEnvironmentTensor{rcd,rt,n,i}(t::AbstractTensorMap) where {rcd,rt,n,i}
        return new{rcd,rt,n,i}(t)
    end
end

mutable struct SparseLeftEnvironmentTensor{N} <: AbstractLeftEnvironmentTensor
    A::Union{Array{<:LeftEnvironmentTensor},Array{<:LeftCompositeEnvironmentTensor}}
    D::NTuple{N,Int}

    function SparseLeftEnvironmentTensor(t::Array{LeftEnvironmentTensor},D::Int64)
        return new{1}(t,(D,))
    end

    function SparseLeftEnvironmentTensor(t::Union{Array{LeftEnvironmentTensor},Array{LeftCompositeEnvironmentTensor}})
        return new{ndims(t)}(t,size(t))
    end

    function SparseLeftEnvironmentTensor(t::LeftEnvironmentTensor)
        return new{1}(convert(Array{LeftEnvironmentTensor},[t]),(1,))
    end

    function SparseLeftEnvironmentTensor(t::AbstractTensorMap)
        return new{1}(convert(Array{LeftEnvironmentTensor},[LeftEnvironmentTensor(t)]),(1,))
    end

    function SparseLeftEnvironmentTensor(t::Array)
        return new{ndims(t)}(convert(Array{LeftEnvironmentTensor},[LeftEnvironmentTensor(ti) for ti in t]),size(t))
    end

    function SparseLeftEnvironmentTensor(::UndefInitializer, n::Integer)
        return new{1}(Vector{LeftEnvironmentTensor}(undef, Int(n)), (Int(n),))
    end

    function SparseLeftEnvironmentTensor(t::LeftEnvironmentTensor, n::Integer)
        n = Int(n)
        A = Vector{LeftEnvironmentTensor}(undef, n)
        T = scalartype(t)
        for i in 1:n
            A[i] = zerovector(t, T)
        end
        return new{1}(A, (n,))
    end

    SparseLeftEnvironmentTensor{N}(A::Union{Array{<:LeftEnvironmentTensor},Array{<:LeftCompositeEnvironmentTensor}}) where N = new{N}(A, size(A))
end

mutable struct SparseRightEnvironmentTensor{N} <: AbstractRightEnvironmentTensor
    A::Union{Array{<:RightEnvironmentTensor},Array{<:RightCompositeEnvironmentTensor}}
    D::NTuple{N,Int}

    function SparseRightEnvironmentTensor(t::Array{RightEnvironmentTensor},D::Int64)
        return new{1}(t,(D,))
    end

    function SparseRightEnvironmentTensor(t::Union{Array{RightEnvironmentTensor},Array{RightCompositeEnvironmentTensor}})
        return new{ndims(t)}(t,size(t))
    end

    function SparseRightEnvironmentTensor(t::RightEnvironmentTensor)
        return new{1}(convert(Array{RightEnvironmentTensor},[t]),(1,))
    end

    function SparseRightEnvironmentTensor(t::AbstractTensorMap)
        return new{1}(convert(Array{RightEnvironmentTensor},[RightEnvironmentTensor(t)]),(1,))
    end

    function SparseRightEnvironmentTensor(t::Array)
        return new{ndims(t)}(convert(Array{RightEnvironmentTensor},[RightEnvironmentTensor(ti) for ti in t]),size(t))
    end

    function SparseRightEnvironmentTensor(::UndefInitializer, n::Integer)
        return new{1}(Vector{RightEnvironmentTensor}(undef, Int(n)), (Int(n),))
    end

    function SparseRightEnvironmentTensor(t::RightEnvironmentTensor, n::Integer)
        n = Int(n)
        A = Vector{RightEnvironmentTensor}(undef, n)
        T = scalartype(t)
        for i in 1:n
            A[i] = zerovector(t, T)
        end
        return new{1}(A, (n,))
    end

    SparseRightEnvironmentTensor{N}(A::Union{Array{<:RightEnvironmentTensor},Array{<:RightCompositeEnvironmentTensor}}) where N = new{N}(A, size(A))
end


mutable struct DenseLeftEnvironmentTensor{R} <: AbstractLeftEnvironmentTensor
    A::LeftEnvironmentTensor

    function DenseLeftEnvironmentTensor(t::AbstractTensorMap)
        return new{numind(t)}(LeftEnvironmentTensor(t))
    end

    function DenseLeftEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{numind(t)}(LeftEnvironmentTensor(t))
    end

    function DenseLeftEnvironmentTensor(t::LeftEnvironmentTensor)
        return new{numind(t.A)}(t)
    end

    function DenseLeftEnvironmentTensor{r}(t::LeftEnvironmentTensor) where r
        return new{r}(t)
    end
end

mutable struct DenseRightEnvironmentTensor{R} <: AbstractLeftEnvironmentTensor
    A::RightEnvironmentTensor

    function DenseRightEnvironmentTensor(t::AbstractTensorMap)
        return new{numind(t)}(RightEnvironmentTensor(t))
    end

    function DenseRightEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{numind(t)}(RightEnvironmentTensor(t))
    end

    function DenseRightEnvironmentTensor(t::RightEnvironmentTensor)
        return new{numind(t.A)}(t)
    end

    function DenseRightEnvironmentTensor{r}(t::RightEnvironmentTensor) where r
        return new{r}(t)
    end
end

Base.adjoint(A::DenseLeftEnvironmentTensor{2}) = DenseLeftEnvironmentTensor(A.A')
Base.adjoint(A::DenseRightEnvironmentTensor{2}) = DenseRightEnvironmentTensor(A.A')
