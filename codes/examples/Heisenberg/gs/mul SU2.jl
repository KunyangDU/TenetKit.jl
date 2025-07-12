using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

# mul!( SparseMPO, MPS) without quantum number
# add quantum number SparseMPO
# 0 -> 1 -> others by isometry
#  done  iso



Lx = 9
Ly = 1


Latt = YCSqua(Lx,Ly)


H = U1Hamiltonian(Latt)

ψ = let 
    AuxSpace = vcat(Rep[SU₂](1//2 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:size(Latt)),], size(Latt)-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end

obs = let 
    Obs = Observable()
    LocalSpace = SU₂Spin

    for i in 1:size(Latt),j in i+1:size(Latt)
        addObs!(Obs,LocalSpace.SS,(i,j),("S","S"),nothing)
    end

    calObs!(Obs, ψ)
end

sum([obs["SS"][i] for i in keys(obs["SS"])])
