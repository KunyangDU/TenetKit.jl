using TensorKit,JLD2,KrylovKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
D = 2^6

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
# t = 1

# Root = InteractionTreeNode()
# LocalSpace = TrivialSpinlessFermion

# for i in 1:size(Latt)
#     addIntr!(Root,LocalSpace.n,i,"n",μ,nothing)
# end

# for pair in neighbor(Latt)
#     addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),-t,LocalSpace.Z)
#     addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),t,LocalSpace.Z)
# end
# Root
H = Hamiltonian(Latt;μ=μ)
lsE,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-6))
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
#A = ψ.ts[1]
#zerovector(ψ.ts[1],Float64)
#@save "examples/TrivialSpinlessFermion/"
