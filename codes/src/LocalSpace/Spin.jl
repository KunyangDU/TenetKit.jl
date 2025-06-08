
module SU₂Spin

using TensorKit

const PhySpace = Rep[SU₂](1//2 => 1)

# S⋅S interaction
const SS = let
    AuxSpace = Rep[SU₂](1 => 1)
    OpL = TensorMap(ones, Float64, PhySpace, AuxSpace ⊗ PhySpace) * sqrt(3) / 2.
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

const S2 = TensorMap(ones, Float64, PhySpace, PhySpace) * 3 / 4
end


module U₁Spin

using TensorKit

const PhySpace = Rep[U₁](1//2 => 1, -1//2 => 1)

const Sz = let 
    Op = TensorMap(ones, PhySpace, PhySpace )
    block(Op, Irrep[U₁](1//2)) .= 1/2
    block(Op, Irrep[U₁](-1//2)) .= -1/2
    Op
end

const SzSz = Sz, Sz

const S₊S₋ = let 
    AuxSpace = Rep[U₁](1 => 1)
    OpL = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

const S₋S₊ = let 
    AuxSpace = Rep[U₁](-1 => 1)
    OpL = TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end
const S2 = TensorMap(ones, Float64, PhySpace, PhySpace) * 3 / 4
const Sz2 = Sz*Sz

end

module U₁Spin1

using TensorKit

const PhySpace = Rep[U₁](1 => 1, 0 => 1, -1 => 1)

const Sz = let 
    Op = TensorMap(zeros, PhySpace, PhySpace )
    block(Op, Irrep[U₁](1)) .= 1
    block(Op, Irrep[U₁](-1)) .= -1
    Op
end

const SzSz = Sz, Sz

const S₊S₋ = let 
    AuxSpace = Rep[U₁](1 => 1)
    OpL = sqrt(2)*TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

const S₋S₊ = let 
    AuxSpace = Rep[U₁](-1 => 1)
    OpL = sqrt(2)*TensorMap(ones, PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end
const S2 = TensorMap(ones, Float64, PhySpace, PhySpace) * 2 
const Sz2 = Sz*Sz

end

module SU₂Spin1

using TensorKit

const PhySpace = Rep[SU₂](1 => 1)

# S⋅S interaction
const SS = let
    AuxSpace = Rep[SU₂](1 => 1)
    OpL = TensorMap(ones, Float64, PhySpace, AuxSpace ⊗ PhySpace) * sqrt(2)
    OpR = permute(OpL', ((2,1), (3,)))
    OpL, OpR
end

const S2 = TensorMap(ones, Float64, PhySpace, PhySpace) * 2
end


