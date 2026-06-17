
module TrivialSpinlessFermion
using TensorKit
const PhySpace = ℂ^2
const Z = TensorMap([1 0; 0 -1],PhySpace,PhySpace)
const F = TensorMap([0 1;0 0],PhySpace,PhySpace)
const F⁺ = TensorMap([0 0;1 0],PhySpace,PhySpace)
const FF⁺ = F, F⁺
const F⁺F = F⁺, F
const n = F⁺*F
const nn = n, n
end

module TrivialSpinfulFermion
using TensorKit
import LinearAlgebra: diagm
const PhySpace = ℂ^4

const Z = TensorMap(diagm([1,-1,-1,1]),PhySpace,PhySpace)
const F₊⁺ = TensorMap(diagm(-2 => [1,1]),PhySpace,PhySpace)
const F₋⁺ = TensorMap(diagm(-1 => [1,0,-1]),PhySpace,PhySpace)
const nd = TensorMap(diagm([0,0,0,2]),PhySpace,PhySpace)
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
