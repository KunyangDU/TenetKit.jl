
module TrivialSpinOneHalf
using TensorKit
import LinearAlgebra: cross

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
const S₊ = TensorMap([0.0 1.0;0.0 0.0],PhySpace,PhySpace)
const S₋ = TensorMap([0.0 0.0;1.0 0.0],PhySpace,PhySpace)
const SxSx = Sx,Sx 
# const SySy = Sy,Sy 
const SySy = TensorMap(-[0 -1;1 0]/2,PhySpace,PhySpace),TensorMap([0 -1;1 0]/2,PhySpace,PhySpace)
const SzSz = Sz,Sz 
const S2 = TensorMap([1 0;0 1]*3/4,PhySpace,PhySpace)

const Sx2 = Sx*Sx 
const Sy2 = Sy*Sy 
const Sz2 = Sz*Sz

const SxSy = Sx,Sy
const SySx = Sy,Sx
const SySz = Sy,Sz
const SzSy = Sz,Sy
const SxSz = Sx,Sz
const SzSx = Sz,Sx
const S₊S₋ = S₊,S₋
const S₋S₊ = S₋,S₊

function _local_axis(h::Vector)
    ŵ = h / norm(h)
    if abs(ŵ[1]) < 0.6 && abs(ŵ[2]) < 0.6
        û = [ŵ[3], 0.0, -ŵ[1]]
    else
        û = [-ŵ[2], ŵ[1], 0.0]
    end
    û /= norm(û)
    v̂ = cross(ŵ,û)
    return û,v̂,ŵ
end

function Sud(h::Vector) 
    x̂,ŷ,_ = _local_axis(h)
    return transpose(x̂ + 1im * ŷ) * [Sx,Sy,Sz] |> x -> (x,x')
end

end

module TrivialSpinOne
using TensorKit
import LinearAlgebra: diagm

const PhySpace = ℂ^3
const Sz = let 
    MatOp = diagm([1,0,-1])
    TensorMap(MatOp,PhySpace,PhySpace)
end
const S₊ = let 
    MatOp = diagm(1 => sqrt(2)*[1,1])
    TensorMap(MatOp,PhySpace,PhySpace)
end

const S₋ = S₊'
const Sx = (S₊ + S₋) / 2
const Sy = (S₊ - S₋) / 2im 
const SxSx = Sx,Sx 
const SySy = Sy,Sy 
const SzSz = Sz,Sz 
const S2 = TensorMap(diagm(ones(3))*2,PhySpace,PhySpace)

const Sx2 = Sx*Sx 
const Sy2 = Sy*Sy 
const Sz2 = Sz*Sz

const Sc = (Sx + Sy + Sz) / sqrt(3)
const Sc2 = Sc*Sc

const SxSy = Sx,Sy
const SySx = Sy,Sx
const SySz = Sy,Sz
const SzSy = Sz,Sy
const SxSz = Sx,Sz
const SzSx = Sz,Sx

end

