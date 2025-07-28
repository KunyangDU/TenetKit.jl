using TensorKit,JLD2,KrylovKit
include("../../src/TenetKit.jl")
include("model.jl")


Lx = 8
Ly = 1
D = 2^6
Latt = YCSqua(Lx,Ly)
@save "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end
t = 1
let 
    Root = InteractionTreeNode()
    LocalSpace = TrivialSpinlessFermion

    # for i in 1:size(Latt)
    #     addIntr!(Root,LocalSpace.n,i,"n",false,μ,nothing)
    # end
    
    # for pair in neighbor(Latt)
    #     addIntr!(Root,LocalSpace.F⁺F,pair,("F⁺","F"),(true,true),-t,LocalSpace.Z)
    #     addIntr!(Root,LocalSpace.FF⁺,pair,("F","F⁺"),(true,true),t,LocalSpace.Z)
    # end
    addIntr!(Root,LocalSpace.F⁺F,(3,6),("F⁺","F"),(true,true),-t,LocalSpace.Z)
    addIntr!(Root,LocalSpace.F⁺,3,"F⁺",true,-t,LocalSpace.Z)

    addIntr3!(Root,(LocalSpace.F⁺,LocalSpace.F⁺,LocalSpace.F⁺),(2,4,6),("F⁺","F","F"),(false,true,true),-t,LocalSpace.Z)
    addIntr4!(Root,(LocalSpace.F⁺,LocalSpace.F⁺,LocalSpace.F⁺,LocalSpace.F⁺),(2,4,6,8),("F⁺","F","F","F"),(true,true,true,true),-t,LocalSpace.Z)

    Root
    # AutomataSparseMPO(Root,size(Latt))
end

# μ = 0
# t = 1
# H = Hamiltonian(Latt;μ=μ)
# lsE,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-12))
# showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))


# @save "examples/TrivialSpinlessFermion/data/ψ_$(Lx)x$(Ly)_$(D).jld2" ψ




