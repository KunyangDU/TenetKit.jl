function _cbeorth!(Q::T,A::T,direction::AbstractDirection) where T <: Union{MPSTensor{<:Number, 3},DenseMPOTensor{<:Number, 4},AdjointMPOTensor{<:Number, 4}}
    Q.A -= _cbeproj(Q,A,direction)
    return norm(_cbeinner(Q,A,direction))
end

function _cbeproj(Q::DenseMPOTensor{<:Number, 4},A::DenseMPOTensor{<:Number, 4},direction::AbstractDirection)
    if typeof(direction) <: L2R
        return @tensor tmp[-1,-2;-3,-4] ≔ Q.A[2,-2,1,3] * A'.A[1,3,2,4] * A.A[-1,4,-3,-4]
    elseif typeof(direction) <: R2L 
        return @tensor tmp[-1,-2;-3,-4] ≔ Q.A[2,1,-3,3] * A'.A[4,3,2,1] * A.A[-1,-2,4,-4]
    end
end

function _cbeinner(Q::DenseMPOTensor{<:Number, 4},A::DenseMPOTensor{<:Number, 4},direction::AbstractDirection)
    if typeof(direction) <: L2R
        return @tensor tmp[-1;-2] ≔ Q.A[2,-1,1,3] * A'.A[1,3,2,-2]
    elseif typeof(direction) <: R2L
        return @tensor tmp[-1;-2] ≔ Q.A[2,1,-2,3] * A'.A[-1,3,2,1]
    end
end

function _cbeproj(Q::MPSTensor{<:Number, 3},A::MPSTensor{<:Number, 3},direction::AbstractDirection)
    if typeof(direction) <: L2R
        return @tensor tmp[-1,-2;-3] ≔ Q.A[-1,2,1] * A'.A[1,3,2] * A.A[3,-2,-3]
    elseif typeof(direction) <: R2L 
        return @tensor tmp[-1,-2;-3] ≔ Q.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3]
    end
end

function _cbeinner(Q::MPSTensor{<:Number, 3},A::MPSTensor{<:Number, 3},direction::AbstractDirection)
    if typeof(direction) <: L2R
        return @tensor tmp[-1;-2] ≔ Q.A[-1,2,1] * A'.A[1,-2,2]
    elseif typeof(direction) <: R2L
        return @tensor tmp[-1;-2] ≔ Q.A[1,2,-2] * A'.A[-1,1,2]
    end
end

_expanddim(::ComplexSpace,D::Int64) = ℂ^D

function _expanddim(S::GradedSpace,D::Int64)
    ratio = D / dim(S)
    for (c,d) in S.dims
        S.dims[c] = ceil(Int64,d*ratio)
    end
    return S
end

function _cbetensor(func,A::MPSTensor{<:Number, 3}, D_f::Int64, direction::AbstractDirection)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if typeof(direction) <: R2L
        tmp = MPSTensor(func,cdm,_expanddim(fuse(cdm),D_f))
    elseif typeof(direction) <: L2R
        tmp = MPSTensor(func,_expanddim(fuse((cdm[2] ⊗ dm)),D_f) ⊗ cdm[2],dm)
    end
    normalize!(tmp)
    return tmp'
end

function _cbetensor(func,A::DenseMPOTensor{<:Number, 4}, D_f::Int64,direction::AbstractDirection)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if typeof(direction) <: R2L
        tmp = DenseMPOTensor(func,cdm,(_expanddim(fuse(cdm⊗dm[2]),D_f) )⊗dm[2])
    elseif typeof(direction) <: L2R
        tmp = DenseMPOTensor(func,cdm[1]⊗(_expanddim(fuse(dm⊗cdm[1]),D_f)),dm)
    end
    normalize!(tmp)
    return tmp'
end

function _cbedsum(Q::MPSTensor{<:Number, 3},A::MPSTensor{<:Number, 3},direction::AbstractDirection)
    if typeof(direction) <: L2R
        ~,Q = rightorth(catcodomain(map(x -> permute(x.A,(1,),(2,3)),(Q,A))...))
        Q = MPSTensor(permute(Q,(1,2),(3,)))
    elseif typeof(direction) <: R2L
        Q,~ = leftorth(catdomain(Q.A,A.A))
        Q = MPSTensor(Q)
    end
    return Q
end

function _cbedsum(Q::DenseMPOTensor{<:Number, 4},A::DenseMPOTensor{<:Number, 4},direction::AbstractDirection)
    if typeof(direction) <: L2R
        Q = catcodomain(map(x -> permute(x.A,(2,),(1,3,4)),(Q,A))...)
        ~,Q = rightorth(DenseMPOTensor(permute(Q,(2,1),(3,4))))
    elseif typeof(direction) <: R2L
        Q = catdomain(map(x -> permute(x.A,(1,2,4),(3,)),(Q,A))...)
        Q,~ = leftorth(DenseMPOTensor(permute(Q,(1,2),(4,3))))
    end
    return Q
end

_cbetensor(func,A::AdjointMPOTensor{<:Number, 4}, D_f::Int64,direction::AbstractDirection) = _cbetensor(func,A',D_f,direction)'
_cbeproj(Q::AdjointMPOTensor{<:Number, 4},A::AdjointMPOTensor{<:Number, 4},direction::AbstractDirection) = _cbeproj(Q',A',direction)'
_cbeinner(Q::AdjointMPOTensor{<:Number, 4},A::AdjointMPOTensor{<:Number, 4},direction::AbstractDirection) = _cbeinner(Q',A',direction)'
_cbedsum(Q::AdjointMPOTensor{<:Number, 4},A::AdjointMPOTensor{<:Number, 4},direction::AbstractDirection) = _cbedsum(Q',A',direction)'

# contract(tl::T, tr::T) where T <: Union{MPSTensor{<:Number, 2},DenseMPOTensor{<:Number, 2}} = tl*tr

# _tdvp_permute(obj::DenseMPOTensor{<:Number, 4}, ::R2L) = DenseMPOTensor(permute(obj.A,(2,1),(3,4)))
# _tdvp_permute(obj::DenseMPOTensor{<:Number, 4}, ::L2R) = DenseMPOTensor(permute(obj.A,(1,2),(4,3)))
# _tdvp_permute(obj::MPSTensor{<:Number, 3}, ::R2L) = MPSTensor(permute(obj.A,(2,1),(3,4)))
# _tdvp_permute(obj::MPSTensor{<:Number, 3}, ::L2R) = MPSTensor(permute(obj.A,(1,2),(4,3)))

function _tdvp_tsvd(tmp::DenseMPOTensor{<:Number, 4},truncsch::TensorKit.TruncationScheme,::R2L)
    localto = TimerOutput()
    @timeit localto "SVD" tl, tc, tr, ϵ = tsvd(tmp; direction=:center,trunc = truncsch, index_tuple = ((2,),(1,3,4)))
    # tr = _tdvp_permute(tr,R2L())
    tr = DenseMPOTensor(permute(tr.A,(2,1),(3,4)))
    @timeit localto "contract" tl = tl*tc
    return tl,tc,tr,ϵ,localto
end

function _tdvp_tsvd(tmp::DenseMPOTensor{<:Number, 4},truncsch::TensorKit.TruncationScheme,::L2R)
    localto = TimerOutput()
    @timeit localto "SVD" tl, tc, tr, ϵ = tsvd(tmp; direction=:center,trunc = truncsch)
    # tl = _tdvp_permute(tl,L2R())
    tl = DenseMPOTensor(permute(tl.A,(1,2),(4,3)))
    @timeit localto "contract" tr = tc*tr
    return tl,tc,tr,ϵ,localto
end

function _tdvp_tsvd(tmp::MPSTensor{<:Number, 3},truncsch::TensorKit.TruncationScheme,::R2L)
    localto = TimerOutput()
    @timeit localto "SVD" tl, tc, tr, ϵ = tsvd(tmp; direction=:center,trunc = truncsch, index_tuple = ((1,),(2,3)))
    tr = MPSTensor(permute(tr.A,(1,2),(3,)))
    @timeit localto "contract" tl = tl*tc
    return tl,tc,tr,ϵ,localto
end

function _tdvp_tsvd(tmp::MPSTensor{<:Number, 3},truncsch::TensorKit.TruncationScheme,::L2R)
    localto = TimerOutput()
    @timeit localto "SVD" tl, tc, tr, ϵ = tsvd(tmp; direction=:center,trunc = truncsch)
    @timeit localto "contract" tr = tc*tr
    return tl,tc,tr,ϵ,localto
end
