mutable struct TruncationScheme
    notrunc::Bool
    truncdim::Int64
    truncbelow::Float64
    TruncationScheme() = new(true,0,0.0)
    TruncationScheme(D::Int64) = new(false,D,0.0)
    TruncationScheme(tol::Float64) = new(false,0,tol)
    TruncationScheme(notrunc::Bool, truncdim::Int64, truncbelow::Float64) = new(notrunc,truncdim,truncbelow)
end
notrunc() = TruncationScheme()
truncdim(D::Int64) = TruncationScheme(D)
truncbelow(tol::Float64) = TruncationScheme(tol)
function Base.getindex(A::TruncationScheme, ::Colon)
    A.notrunc && return (;)
    A.truncbelow == 0.0 && A.truncdim == 0 && return (;)
    A.truncbelow > 0.0 && A.truncdim == 0 && return (atol = A.truncbelow,)
    A.truncbelow == 0.0 && A.truncdim > 0 && return (maxrank = A.truncdim,) 
    A.truncbelow > 0.0 && A.truncdim > 0 && return (atol = A.truncbelow, maxrank = A.truncdim) 
end

_getdim(trunc::TruncationScheme) = trunc.truncdim
_getcutoff(trunc::TruncationScheme) = trunc.truncbelow
# _getcutoff(trunc::TruncationScheme) = trunc.truncbelow == 0.0 ? nothing : trunc.truncbelow 
# _!getdim(trunc::TruncationScheme) = filter(x -> typeof(x) != TensorKit.TruncationDimension, collect(trunc.truncations))
# function _updatedim(trunc::TruncationScheme,ratio::Number)
#     trunc.truncdim = ceil(Int64,trunc.truncdim*ratio)
#     return trunc
# end

# _getdim(trunc::TruncationScheme) = trunc.truncdim
truncdim(trunc::TruncationScheme,ratio::Number) = TruncationScheme(ceil(Int64,trunc.truncdim*ratio))
truncdim(trunc::TruncationScheme) = TruncationScheme(trunc.truncdim)

Base.:&(A::TruncationScheme, B::TruncationScheme) = TruncationScheme(A.notrunc & B.notrunc, max(A.truncdim,B.truncdim), max(A.truncbelow,B.truncbelow))
