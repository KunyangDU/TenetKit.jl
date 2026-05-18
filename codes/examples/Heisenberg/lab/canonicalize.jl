using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")

D = 1
Lx = 8
Ly = 1
params = (J=1, Δ = 1)

Latt = YCSqua(Lx,Ly)

ψ = let 
    AuxSpace = repeat([ℂ^D,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace, isdisk = true)
end
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

ψ1 = ψ'
ρ1 = ρ'
canonicalize!(ψ1,4)
canonicalize!(ρ1,1)
1