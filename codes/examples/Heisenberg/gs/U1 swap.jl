using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/U1"

D = 2^6
params = (Jz = 1,Jxy = 0.5)

Lx = 10
Ly = 1
Latt = YCSqua(Lx,Ly)

H = let LocalSpace = U₁Spin
    Jz = params.Jz
    Jxy = params.Jxy
    Root = InteractionTreeNode()

    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),Jz,nothing)
    end

    for (i,j) in neighbor(Latt;ordered = true)
        addIntr!(Root,LocalSpace.S₊S₋,(i,j),("S₊","S₋"),Jxy,nothing)
    end

    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end



ψ = let 
    AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt)//2 :1//2:size(Latt)//2 ),], Lx*Ly-1))
    randMPS(U₁Spin.PhySpace ,AuxSpace)
end

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg / size(Latt) .- 1/4)



