using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/U1"




D = 2^6
J = 1

Lx = 10
Ly = 1
Latt = YCSqua(Lx,Ly)

H = let LocalSpace = SU₂Spin
    Root = InteractionTreeNode()

    for (i,j) in neighbor(Latt;ordered = true)
        if i < j
            addIntr!(Root,LocalSpace.SS,(i,j),("S","S"),J,nothing)
        else
            # addIntr!(Root,swap(LocalSpace.SS),(j,i),("S","S"),J,nothing)
        end
    end

    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end



ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], size(Latt)-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg / size(Latt) .- 1/4)



