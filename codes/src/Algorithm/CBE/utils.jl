function _cbeorth!(Q::T,A::T,direction::AbstractDirection) where T <: Union{MPSTensor{3},DenseMPOTensor{4},AdjointMPOTensor{4}}
    Q.A -= _cbeproj(Q,A,direction)
    return norm(_cbeinner(Q,A,direction))
end

function _cbeproj(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::AbstractDirection)
    if typeof(direction) <: L2R
        return @tensor tmp[-1,-2;-3,-4] ≔ Q.A[2,-2,1,3] * A'.A[1,3,2,4] * A.A[-1,4,-3,-4]
    elseif typeof(direction) <: R2L 
        return @tensor tmp[-1,-2;-3,-4] ≔ Q.A[2,1,-3,3] * A'.A[4,3,2,1] * A.A[-1,-2,4,-4]
    end
end

function _cbeinner(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::AbstractDirection)
    if typeof(direction) <: L2R
        return @tensor tmp[-1;-2] ≔ Q.A[2,-1,1,3] * A'.A[1,3,2,-2]
    elseif typeof(direction) <: R2L
        return @tensor tmp[-1;-2] ≔ Q.A[2,1,-2,3] * A'.A[-1,3,2,1]
    end
end

function _cbeproj(Q::MPSTensor{3},A::MPSTensor{3},direction::AbstractDirection)
    if typeof(direction) <: L2R
        return @tensor tmp[-1,-2;-3] ≔ Q.A[-1,2,1] * A'.A[1,3,2] * A.A[3,-2,-3]
    elseif typeof(direction) <: R2L 
        return @tensor tmp[-1,-2;-3] ≔ Q.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3]
    end
end

function _cbeinner(Q::MPSTensor{3},A::MPSTensor{3},direction::AbstractDirection)
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

function _cbetensor(func,A::MPSTensor{3}, D_f::Int64, direction::AbstractDirection)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if typeof(direction) <: R2L
        tmp = MPSTensor(func,cdm,_expanddim(fuse(cdm),D_f))
    elseif typeof(direction) <: L2R
        tmp = MPSTensor(func,_expanddim(fuse((cdm[2] ⊗ dm)),D_f) ⊗ cdm[2],dm)
    end
    normalize!(tmp)
    return tmp'
end

function _cbetensor(func,A::DenseMPOTensor{4}, D_f::Int64,direction::AbstractDirection)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if typeof(direction) <: R2L
        tmp = DenseMPOTensor(func,cdm,(_expanddim(fuse(cdm⊗dm[2]),D_f) )⊗dm[2])
    elseif typeof(direction) <: L2R
        tmp = DenseMPOTensor(func,cdm[1]⊗(_expanddim(fuse(dm⊗cdm[1]),D_f)),dm)
    end
    normalize!(tmp)
    return tmp'
end

# function _cbedsum(Q::MPSTensor{3},A::MPSTensor{3},direction::AbstractDirection)
#     if typeof(direction) <: L2R
#         # ~,Q = rightorth(catcodomain(map(x -> permute(x.A,((1,),(2,3))),(Q,A))...))
#         Q = catcodomain(map(x -> permute(x.A,((1,),(2,3))),(Q,A))...)
#         Q = MPSTensor(permute(Q, ((1, 2), (3,))))
#     elseif typeof(direction) <: R2L
#         # Q,~ = leftorth(catdomain(Q.A,A.A))
#         Q = catdomain(Q.A,A.A)
#         Q = MPSTensor(Q)
#     end
#     return Q
# end

# function _cbedsum(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::AbstractDirection)
#     if typeof(direction) <: L2R
#         Q = catcodomain(map(x -> permute(x.A,((2,),(1,3,4))),(Q,A))...)
#         # ~,Q = rightorth(DenseMPOTensor(permute(Q, ((2, 1), (3,4)))))
#         Q = DenseMPOTensor(permute(Q, ((2, 1), (3,4))))
#     elseif typeof(direction) <: R2L
#         Q = catdomain(map(x -> permute(x.A, ((1, 2,4), (3,))),(Q,A))...)
#         # Q,~ = leftorth(DenseMPOTensor(permute(Q, ((1, 2), (4,3)))))
#         Q = DenseMPOTensor(permute(Q, ((1, 2), (4,3))))
#     end
#     return Q
# end

# function _cbedsum(Q::MPSTensor{3},A::MPSTensor{3},::L2R)
#     # ~,Q = rightorth(catcodomain(map(x -> permute(x.A,((1,),(2,3))),(Q,A))...))
#     Q = catcodomain(map(x -> permute(x.A,((1,),(2,3))),(Q,A))...)
#     Q = MPSTensor(permute(Q, ((1, 2), (3,))))
# end 

# function _cbedsum(Q::MPSTensor{3},A::MPSTensor{3},::R2L)
#     # Q,~ = leftorth(catdomain(Q.A,A.A))
#     Q = catdomain(Q.A,A.A)
#     Q = MPSTensor(Q)
#     return Q
# end

# function _cbedsum(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},::L2R)
#     Q = catcodomain(map(x -> permute(x.A,((2,),(1,3,4))),(Q,A))...)
#     # ~,Q = rightorth(DenseMPOTensor(permute(Q, ((2, 1), (3,4)))))
#     Q = DenseMPOTensor(permute(Q, ((2, 1), (3,4))))
#     return Q
# end

# function _cbedsum(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},::R2L)
#     Q = catdomain(map(x -> permute(x.A, ((1, 2,4), (3,))),(Q,A))...)
#     # Q,~ = leftorth(DenseMPOTensor(permute(Q, ((1, 2), (4,3)))))
#     Q = DenseMPOTensor(permute(Q, ((1, 2), (4,3))))
#     return Q
# end


function _loplus(A::MPSTensor{3},B::MPSTensor{3})
    Q = catcodomain(map(x -> permute(x.A,((1,),(2,3))),(A,B))...)
    Q = MPSTensor(permute(Q, ((1, 2), (3,))))
    return Q
end

function _roplus(A::MPSTensor{3}, B::MPSTensor{3})
    Q = catdomain(A.A,B.A)
    Q = MPSTensor(Q)
end

function _loplus(A::DenseMPOTensor{4},B::DenseMPOTensor{4})
    Q = catcodomain(map(x -> permute(x.A,((2,),(1,3,4))),(A,B))...)
    Q = DenseMPOTensor(permute(Q, ((2, 1), (3,4))))
    return Q
end

function _roplus(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    Q = catdomain(map(x -> permute(x.A, ((1, 2,4), (3,))),(A,B))...)
    Q = DenseMPOTensor(permute(Q, ((1, 2), (4,3))))
end

function _lexpand(Al::DenseMPOTensor{4}, Ar::DenseMPOTensor{4})
    return Al, DenseMPOTensor(zeros(eltype(Ar), codomain(Ar)[1] ⊗ domain(Al)[1], domain(Ar)))
end

function _rexpand(Al::DenseMPOTensor{4}, Ar::DenseMPOTensor{4})
    return DenseMPOTensor(zeros(eltype(Al), codomain(Al), codomain(Ar)[2] ⊗ domain(Al)[2])), Ar
end

function _lexpand(Al::MPSTensor{3}, Ar::MPSTensor{3})
    return Al, MPSTensor(zeros(eltype(Ar), domain(Al)[1] ⊗ codomain(Ar)[2], domain(Ar)))
end

function _rexpand(Al::MPSTensor{3}, Ar::MPSTensor{3})
    return MPSTensor(zeros(eltype(Al), codomain(Al), codomain(Ar)[1])), Ar
end

_cbetensor(func,A::AdjointMPOTensor{4}, D_f::Int64,direction::AbstractDirection) = _cbetensor(func,A',D_f,direction)'
_cbeproj(Q::AdjointMPOTensor{4},A::AdjointMPOTensor{4},direction::AbstractDirection) = _cbeproj(Q',A',direction)'
_cbeinner(Q::AdjointMPOTensor{4},A::AdjointMPOTensor{4},direction::AbstractDirection) = _cbeinner(Q',A',direction)'
_cbedsum(Q::AdjointMPOTensor{4},A::AdjointMPOTensor{4},direction::AbstractDirection) = _cbedsum(Q',A',direction)'


_cbe_maxdim(env::Environment{3}, ::CBEalgo{sch,struc,1}, ::CBEinfo{L2R}) where {sch,struc} = (site = env.center[1]; return _cbe_maxdim(env.layer[1][site:site+1]...))
_cbe_maxdim(env::Environment{3}, ::CBEalgo{sch,struc,1}, ::CBEinfo{R2L}) where {sch,struc} = (site = env.center[1]; return _cbe_maxdim(env.layer[1][site-1:site]...))

_cbe_maxdim(env::Environment{3}, ::CBEalgo{sch,struc,3}, ::CBEinfo{L2R}) where {sch,struc} = (site = env.center[1]; return _cbe_maxdim(env.layer[3][site:site+1]...))
_cbe_maxdim(env::Environment{3}, ::CBEalgo{sch,struc,3}, ::CBEinfo{R2L}) where {sch,struc} = (site = env.center[1]; return _cbe_maxdim(env.layer[3][site-1:site]...))


function _cbe_maxdim(Al::DenseMPOTensor{4}, Ar::DenseMPOTensor{4})
    Dl = mapreduce(i -> dim(Al.A, i)[2], *, [1,2,4])
    Dr = mapreduce(i -> dim(Ar.A, i)[2], *, [1,3,4])
    return Dl,Dr
end

function _cbe_maxdim(Al::AdjointMPOTensor{4}, Ar::AdjointMPOTensor{4})
    Dl = mapreduce(i -> dim(Al.A, i)[2], *, [2,3,4])
    Dr = mapreduce(i -> dim(Ar.A, i)[2], *, [1,2,3])
    return Dl,Dr
end

function _cbe_maxdim(Al::MPSTensor{3}, Ar::MPSTensor{3})
    Dl = mapreduce(i -> dim(Al.A, i)[2], *, [1,2])
    Dr = mapreduce(i -> dim(Ar.A, i)[2], *, [2,3])
    return Dl,Dr
end

function _cbe_maxdim(Al::AdjointMPSTensor{3}, Ar::AdjointMPSTensor{3})
    Dl = mapreduce(i -> dim(Al.A, i)[2], *, [2,3])
    Dr = mapreduce(i -> dim(Ar.A, i)[2], *, [1,3])
    return Dl,Dr
end

_cbe_currentdim(env::Environment{3}, ::CBEalgo{sch,struc,1}, ::CBEinfo{L2R}) where {sch,struc} = (site = env.center[1]; return _cbe_currentdim(env.layer[1][site:site+1]...))
_cbe_currentdim(env::Environment{3}, ::CBEalgo{sch,struc,1}, ::CBEinfo{R2L}) where {sch,struc} = (site = env.center[1]; return _cbe_currentdim(env.layer[1][site-1:site]...))

_cbe_currentdim(env::Environment{3}, ::CBEalgo{sch,struc,3}, ::CBEinfo{L2R}) where {sch,struc} = (site = env.center[1]; return _cbe_currentdim(env.layer[3][site:site+1]...))
_cbe_currentdim(env::Environment{3}, ::CBEalgo{sch,struc,3}, ::CBEinfo{R2L}) where {sch,struc} = (site = env.center[1]; return _cbe_currentdim(env.layer[3][site-1:site]...))

_cbe_currentdim(::DenseMPOTensor{4}, Ar::DenseMPOTensor{4}) = dim(Ar.A, 2)[2]
_cbe_currentdim(::AdjointMPOTensor{4}, Ar::AdjointMPOTensor{4}) = dim(Ar.A, 4)[2]
_cbe_currentdim(::MPSTensor{3}, Ar::MPSTensor{3}) = dim(Ar.A, 1)[2]
_cbe_currentdim(::AdjointMPSTensor{3}, Ar::AdjointMPSTensor{3}) = dim(Ar.A, 2)[2]

