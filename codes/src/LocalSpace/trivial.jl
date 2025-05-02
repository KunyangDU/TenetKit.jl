
module TrivialSpinlessFermion
using TensorKit
const PhySpace = ℂ^2
const Z = TensorMap([-1 0; 0 1],PhySpace,PhySpace)
const F = TensorMap([0 0;1 0],PhySpace,PhySpace)
const F⁺ = TensorMap([0 1;0 0],PhySpace,PhySpace)
const FF⁺ = F, F⁺
const F⁺F = F⁺, F
const n = F⁺*F
const nn = n, n
end

module TrivialSpinfulFermion
using TensorKit
function diagm(dg::Vector{T}) where T
    L = length(dg)
    mat = zeros(T,L,L)
    for (dgi,dge) in enumerate(dg)
        mat[dgi,dgi] = dge
    end
    return mat
end
const PhySpace = ℂ^4
function diagm(pair::Pair{Int64, Vector{T}}) where T
    L = length(pair[2]) + abs(pair[1])
    mat = zeros(T,L,L)
    if pair[1] > 0
        for (ii,ie) in enumerate(pair[2])
            mat[ii,ii+pair[1]] = ie
        end
    elseif pair[1] < 0
        for (ii,ie) in enumerate(pair[2])
            mat[ii-pair[1],ii] = ie
        end
    else
        mat = diagm(pair[2])
    end
    
    return mat
end
const Z = TensorMap(diagm([1,-1,-1,1]),PhySpace,PhySpace)
const F₊⁺ = TensorMap(diagm(2 => [1,1]),PhySpace,PhySpace)
const F₋⁺ = TensorMap(diagm(1 => [1,0,1]),PhySpace,PhySpace)
const nd = TensorMap(diagm([2,0,0,0]),PhySpace,PhySpace)
const F₊ = F₊⁺'
const F₋ = F₋⁺'
const F₊⁺F₊ = F₊⁺, F₊
const F₊F₊⁺ = F₊, F₊⁺
const F₋⁺F₋ = F₋⁺, F₋
const F₋F₋⁺ = F₋, F₋⁺
const n₊ = F₊⁺*F₊
const n₋ = F₋⁺*F₋
const n = F₊⁺*F₊ + F₋⁺*F₋
const nn = n, n
end

module TrivialSpinOneHalf
using TensorKit
const PhySpace = ℂ^2
const Sx = let 
    MatOp = [0 1;1 0] / 2
    TensorMap(MatOp,PhySpace,PhySpace)
end
const Sy = let 
    MatOp = [0 -1im;+1im 0] / 2
    TensorMap(MatOp,PhySpace,PhySpace)
end
const Sz = let 
    MatOp = [1 0; 0 -1] / 2
    TensorMap(MatOp,PhySpace,PhySpace)
end
const S₊ = Sx + 1im * Sy
const S₋ = S₊'
const SxSx = Sx,Sx 
const SySy = Sy,Sy 
const SzSz = Sz,Sz 
const S2 = TensorMap([1 0;0 1]*3/4,PhySpace,PhySpace)
const Sz2 = Sz*Sz

const SxSy = Sx,Sy
const SySx = Sy,Sx
const SySz = Sy,Sz
const SzSy = Sz,Sy
const SxSz = Sx,Sz
const SzSx = Sz,Sx
end

