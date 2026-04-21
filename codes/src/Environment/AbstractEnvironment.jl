


mutable struct LocalEnvironmentTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractEnvironmentTensor
    A::T

    function LocalEnvironmentTensor(t::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(t), rank(t), TM}(t)
    end
    function LocalEnvironmentTensor{S}(t::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, rank(t), TM}(t)
    end
    function LocalEnvironmentTensor{S, R, TM}(t::TM) where {S<:Number, R, TM<:AbstractTensorMap}
        return new{S, R, TM}(t)
    end
end

mutable struct RightEnvironmentTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractEnvironmentTensor
    A::T

    function RightEnvironmentTensor(t::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(t), rank(t), TM}(t)
    end
    function RightEnvironmentTensor{S}(t::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, rank(t), TM}(t)
    end
    function RightEnvironmentTensor{S, R, TM}(t::TM) where {S<:Number, R, TM<:AbstractTensorMap}
        return new{S, R, TM}(t)
    end
end

mutable struct LeftEnvironmentTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractEnvironmentTensor
    A::T

    function LeftEnvironmentTensor(t::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(t), rank(t), TM}(t)
    end
    function LeftEnvironmentTensor{S}(t::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, rank(t), TM}(t)
    end
    function LeftEnvironmentTensor{S, R, TM}(t::TM) where {S<:Number, R, TM<:AbstractTensorMap}
        return new{S, R, TM}(t)
    end
end

Base.adjoint(A::LeftEnvironmentTensor{<:Number, 2})  = LeftEnvironmentTensor(A.A')
Base.adjoint(A::RightEnvironmentTensor{<:Number, 2}) = RightEnvironmentTensor(A.A')

# function Base.adjoint(A::LeftEnvironmentTensor{<:Number, 3})
#     tmp = A.A'
#     LeftEnvironmentTensor{<:Number, 3}(permute(tmp,(2,),(1,3)))
# end

# function Base.adjoint(A::RightEnvironmentTensor{<:Number, 3})
#     tmp = A.A'
#     RightEnvironmentTensor{<:Number, 3}(permute(tmp,(1,),(3,2)))
# end

"""

"""
mutable struct LeftCompositeEnvironmentTensor{S<:Number, Rcd,Rt,N,I, T<:AbstractTensorMap} <: AbstractEnvironmentTensor
    A::T

    function LeftCompositeEnvironmentTensor(t::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(t), length(codomain(t)), rank(t), 3, 3, TM}(t)
    end

    function LeftCompositeEnvironmentTensor{S}(t::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, length(codomain(t)), rank(t), 3, 3, TM}(t)
    end

    function LeftCompositeEnvironmentTensor(t::TM, n::Int64, i::Int64) where {TM<:AbstractTensorMap}
        return new{scalartype(t), length(codomain(t)), rank(t), n, i, TM}(t)
    end

    function LeftCompositeEnvironmentTensor{S, Rcd, Rt, N, I, TM}(t::TM) where {S<:Number, Rcd, Rt, N, I, TM<:AbstractTensorMap}
        return new{S, Rcd, Rt, N, I, TM}(t)
    end
end

mutable struct RightCompositeEnvironmentTensor{S<:Number, Rcd,Rt,N,I, T<:AbstractTensorMap} <: AbstractEnvironmentTensor
    A::T

    function RightCompositeEnvironmentTensor(t::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(t), length(domain(t)), rank(t), 3, 3, TM}(t)
    end

    function RightCompositeEnvironmentTensor{S}(t::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, length(domain(t)), rank(t), 3, 3, TM}(t)
    end

    function RightCompositeEnvironmentTensor(t::TM, n::Int64, i::Int64) where {TM<:AbstractTensorMap}
        return new{scalartype(t), length(domain(t)), rank(t), n, i, TM}(t)
    end

    function RightCompositeEnvironmentTensor{S, Rcd, Rt, N, I, TM}(t::TM) where {S<:Number, Rcd, Rt, N, I, TM<:AbstractTensorMap}
        return new{S, Rcd, Rt, N, I, TM}(t)
    end
end

mutable struct SparseEnvironmentTensor <: AbstractEnvironmentTensor
    A::Vector{AbstractEnvironmentTensor}
    D::Int64

    function SparseEnvironmentTensor(t::Vector{AbstractEnvironmentTensor},D::Int64)
        return new(t,D)
    end

    function SparseEnvironmentTensor(t::Vector{AbstractEnvironmentTensor})
        return new(t,length(t))
    end

    function SparseEnvironmentTensor(t::Union{LeftEnvironmentTensor,RightEnvironmentTensor})
        return new(convert(Vector{AbstractEnvironmentTensor},[t]),1)
    end
end



mutable struct SparseLeftEnvironmentTensor{N} <: AbstractLeftEnvironmentTensor
    A::Union{Array{LeftEnvironmentTensor},Array{LeftCompositeEnvironmentTensor}}
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
end

mutable struct SparseRightEnvironmentTensor{N} <: AbstractRightEnvironmentTensor
    A::Union{Array{RightEnvironmentTensor},Array{RightCompositeEnvironmentTensor}}
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
end


mutable struct DenseLeftEnvironmentTensor{R, ET<:LeftEnvironmentTensor} <: AbstractLeftEnvironmentTensor
    A::ET

    function DenseLeftEnvironmentTensor(t::AbstractTensorMap)
        et = LeftEnvironmentTensor(t)
        return new{rank(t), typeof(et)}(et)
    end

    function DenseLeftEnvironmentTensor{r}(t::AbstractTensorMap) where r
        et = LeftEnvironmentTensor(t)
        return new{rank(t), typeof(et)}(et)
    end

    function DenseLeftEnvironmentTensor(t::ET) where {ET<:LeftEnvironmentTensor}
        return new{rank(t.A), ET}(t)
    end

    function DenseLeftEnvironmentTensor{R,ET}(t::ET) where {R, ET<:LeftEnvironmentTensor}
        return new{R, ET}(t)
    end
end

mutable struct DenseRightEnvironmentTensor{R, ET<:RightEnvironmentTensor} <: AbstractLeftEnvironmentTensor
    A::ET

    function DenseRightEnvironmentTensor(t::AbstractTensorMap)
        et = RightEnvironmentTensor(t)
        return new{rank(t), typeof(et)}(et)
    end

    function DenseRightEnvironmentTensor{r}(t::AbstractTensorMap) where r
        et = RightEnvironmentTensor(t)
        return new{rank(t), typeof(et)}(et)
    end

    function DenseRightEnvironmentTensor(t::ET) where {ET<:RightEnvironmentTensor}
        return new{rank(t.A), ET}(t)
    end

    function DenseRightEnvironmentTensor{R,ET}(t::ET) where {R, ET<:RightEnvironmentTensor}
        return new{R, ET}(t)
    end
end

Base.adjoint(A::DenseLeftEnvironmentTensor{2})  = DenseLeftEnvironmentTensor(A.A')
Base.adjoint(A::DenseRightEnvironmentTensor{2}) = DenseRightEnvironmentTensor(A.A')

"""
Monolayer Environment, i.e., only one layer MPO is considered.
"""
mutable struct Environment{N, L, LayerT<:Tuple} <: AbstractEnvironment
    layer::LayerT                                        # typed Tuple → concrete element types
    envs::Union{Nothing,Array{AbstractEnvironmentTensor}}
    center::Vector{Int64}
    L::Int64

    # Constructor from a Vector (user-facing API unchanged): convert → Tuple for type stability
    function Environment(layer_vec::AbstractVector,
        envs::Array{AbstractEnvironmentTensor},
        center::Union{Nothing,Vector{Int64}},
        L::Union{Nothing,Int64})
        lt = Tuple(layer_vec)
        N  = length(lt)
        Lv = something(L, length(lt[1]))
        return new{N, Lv, typeof(lt)}(lt, envs, center, Lv)
    end

    function Environment(layer_vec::AbstractVector)
        lt = Tuple(layer_vec)
        N  = length(lt)
        Lv = length(lt[1])
        return new{N, Lv, typeof(lt)}(lt, nothing, [1,Lv], Lv)
    end

end

mutable struct CBEenvironment <: AbstractEnvironment
    tL₀::AbstractTensorWrapper
    tR₀::AbstractTensorWrapper
    tL::Union{AbstractTensorWrapper,Nothing}
    tR::Union{AbstractTensorWrapper,Nothing}
    D_i::Int64
    D_f::Int64
    Λ::Union{AbstractTensorWrapper,Nothing}
    Lorth::Union{SparseLeftEnvironmentTensor,LeftCompositeEnvironmentTensor,LeftEnvironmentTensor,DenseLeftEnvironmentTensor,Nothing}
    Rorth::Union{SparseRightEnvironmentTensor,RightCompositeEnvironmentTensor,RightEnvironmentTensor,DenseRightEnvironmentTensor,Nothing}

    # function CBEenvironment(env::Environment{N,L},ind::Int64,D_i::Int64,D_f::Int64) where {N,L}
    #     return new(env,nothing,nothing,nothing,ind,D_i,D_f,N,L)
    # end

    # function CBEenvironment(
    #     Λ::AbstractTensorWrapper,
    #     tl₀::AbstractTensorWrapper,
    #     tr₀::AbstractTensorWrapper,
    #     tl::AbstractTensorWrapper,
    #     tr::AbstractTensorWrapper,
    #     Lorth::LeftCompositeEnvironmentTensor,
    #     Rorth::RightCompositeEnvironmentTensor,
    #     D_i::Int64,
    #     D_f::Int64,
    # )
    # return new(Λ,tl₀,tr₀,tl,tr,Lorth,Rorth,D_i,D_f)
    # end
end

# function _fullCBEenv(tL₀::AbstractTensorWrapper,tR₀::AbstractTensorWrapper,D_i::Int64,D_f::Int64)
#     return CBEenvironment(tL₀,tR₀,nothing,nothing,D_i,D_f,nothing,nothing,nothing)
# end

# function _randCBEenv(
#     tL₀::AbstractTensorWrapper,tR₀::AbstractTensorWrapper,
#     tL::Union{AbstractTensorWrapper,Nothing},tR::Union{AbstractTensorWrapper,Nothing},
#     D_i::Int64,D_f::Int64,Λ::AbstractTensorWrapper,
#     Lorth::Union{SparseLeftEnvironmentTensor,LeftCompositeEnvironmentTensor,LeftEnvironmentTensor},
#     Rorth::Union{SparseRightEnvironmentTensor,RightCompositeEnvironmentTensor,RightEnvironmentTensor})
#     return CBEenvironment(tL₀,tR₀,tL,tR,D_i,D_f,Λ,Lorth,Rorth)
# end

