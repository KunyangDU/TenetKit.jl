module U₁SU₂Fermion

using TensorKit

const PhySpace = Rep[U₁×SU₂]((-1,0) => 1, (0,1/2) => 1, (1,0) => 1)
    
const Z = let 
    tmp = ones(PhySpace,PhySpace)
    block(tmp,Irrep[U₁×SU₂](0,1/2)) .= -1
    tmp
end

const n = let 
    tmp = zeros(PhySpace,PhySpace)
    block(tmp,Irrep[U₁×SU₂](0,1/2)) .= 1
    block(tmp,Irrep[U₁×SU₂](1,0)) .= 2
    tmp
end

const nd = let 
    tmp = zeros(PhySpace,PhySpace)
    block(tmp,Irrep[U₁×SU₂](1,0)) .= 1
    tmp
end

const F⁺F = let 
    AuxSpace = Rep[U₁×SU₂]((1,1/2) => 1)
    F⁺ = ones( PhySpace, AuxSpace ⊗ PhySpace)
    block(F⁺, Irrep[U₁×SU₂](1, 0)) .= -sqrt(2)
    F = permute(F⁺', ((2, 1), (3,)))
    block(F, Irrep[U₁×SU₂](1, 0)) .= sqrt(2)
    F⁺, F
end

const FF⁺ = let 
    AuxSpace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
    rev = isometry(AuxSpace, flip(AuxSpace))
    @tensor F[-1; -2 -3] ≔ F⁺F[1]'[1,-1,-3] * rev'[-2,1]
    @tensor F⁺[-1 -2; -3] ≔ F⁺F[2]'[-1,-3,1] * rev[1,-2]
    F, F⁺
end

const SS = let
    AuxSpace = Rep[U₁×SU₂]((0,1) => 1)
    OpL = ones( Float64, PhySpace, AuxSpace ⊗ PhySpace) * sqrt(3) / 2.
    OpR = permute(OpL', ((2, 1), (3,)))
    OpL, OpR
end

end


module U₁U₁Fermion

using TensorKit

const PhySpace = Rep[U₁×U₁]((-1,0) => 1, (0,1 // 2) => 1, (0,-1 // 2) => 1,(1,0) => 1)
    
const Z = let 
    tmp = ones(PhySpace,PhySpace)
    block(tmp,Irrep[U₁×U₁](0,1/2)) .= -1
    block(tmp,Irrep[U₁×U₁](0,-1/2)) .= -1
    tmp
end

const n₊ = let 
    tmp = zeros(PhySpace,PhySpace)
    block(tmp,Irrep[U₁×U₁](0,1/2)) .= 1
    block(tmp,Irrep[U₁×U₁](1,0)) .= 1
    tmp
end

const n₋ = let 
    tmp = zeros(PhySpace,PhySpace)
    block(tmp,Irrep[U₁×U₁](0,-1/2)) .= 1
    block(tmp,Irrep[U₁×U₁](1,0)) .= 1
    tmp
end

const n = n₊ + n₋

const nd = n₊ * n₋

const Sz = (n₊ - n₋) / 2

const F₊⁺F₊ = let
    AuxSpace = Rep[U₁×U₁]((1,1/2) => 1)
    F⁺ = ones( PhySpace, AuxSpace ⊗ PhySpace)
    F = ones( PhySpace ⊗ AuxSpace, PhySpace)
    F⁺, F
end

const F₋⁺F₋ = let
    AuxSpace = Rep[U₁×U₁]((1,-1/2) => 1)
    F⁺ = ones( PhySpace, AuxSpace ⊗ PhySpace)
    F = ones( PhySpace ⊗ AuxSpace, PhySpace)
    block(F⁺, Irrep[U₁×U₁](1, 0)) .= -1
    block(F, Irrep[U₁×U₁](1, 0)) .= -1
    F⁺, F
end

const F₊F₊⁺ = let 
    AuxSpace = Rep[U₁×U₁]((1, 1 / 2) => 1)
    rev = isometry(AuxSpace, flip(AuxSpace))
    @tensor F[-1;-2 -3] ≔ F₊⁺F₊[1]'[1,-1,-3] * rev'[-2,1]
    @tensor F⁺[-1 -2; -3] ≔ F₊⁺F₊[2]'[-1,-3,1] * rev[1,-2]
    -F, -F⁺
end

const F₋F₋⁺ = let 
    AuxSpace = Rep[U₁×U₁]((1, -1 / 2) => 1)
    rev = isometry(AuxSpace, flip(AuxSpace))
    @tensor F[-1; -2 -3] ≔ F₋⁺F₋[1]'[1,-1,-3] * rev'[-2,1]
    @tensor F⁺[-1 -2; -3] ≔ F₋⁺F₋[2]'[-1,-3,1] * rev[1,-2]
    -F, -F⁺
end

const S₊S₋ = let 
    AuxSpace = Rep[U₁×U₁]((0,1) => 1)
    OpL = ones( PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2, 1), (3,)))
    OpL, OpR
end

const S₋S₊ = let 
    AuxSpace = Rep[U₁×U₁]((0,-1) => 1)
    OpL = ones( PhySpace, AuxSpace ⊗ PhySpace)
    OpR = permute(OpL', ((2, 1), (3,)))
    OpL, OpR
end

end


module U₁Fermion 

using TensorKit

const PhySpace = Rep[U₁](-1//2 => 1, 1//2 => 1)

const Z = let 
    tmp = zeros(PhySpace,PhySpace)
    block(tmp,Irrep[U₁](1//2)) .= -1
    block(tmp,Irrep[U₁](-1//2)) .= 1
    tmp
end

const n = let 
    tmp = zeros(PhySpace,PhySpace)
    block(tmp,Irrep[U₁](1//2)) .= 1
    tmp
end

const F⁺F = let 
    AuxSpace = Rep[U₁](1 => 1)
    F⁺ = ones( PhySpace, AuxSpace ⊗ PhySpace )
    F = permute(F⁺', ((2, 1), (3,)))
    F⁺, F
end

const FF⁺ = let 
#=     AuxSpace = Rep[U₁](-1 => 1)
    F = ones( PhySpace, AuxSpace ⊗ PhySpace)
    F⁺ = permute(F', ((2, 1), (3,))) =#
    AuxSpace = Rep[U₁](1 => 1)
    rev = isometry(AuxSpace, flip(AuxSpace))
    @tensor F[-1; -2 -3] ≔ F⁺F[1]'[1,-1,-3] * rev'[-2,1]
    @tensor F⁺[-1 -2; -3] ≔ F⁺F[2]'[-1,-3,1] * rev[1,-2]
    F, F⁺
end

end


