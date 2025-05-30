using TensorKit,AbstractTrees
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/SU2"

# abstract type AbstractTreeNode end
Lx = 8
Ly = 8
Latt = YCSqua(Lx,Ly)

Obs = Observable()
LocalSpace = SU₂Spin

for i in 1:size(Latt), j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SS,(i,j),("S","S"),nothing)
end

ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], size(Latt)-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end
calObs!(Obs,ψ)
Obs.values





