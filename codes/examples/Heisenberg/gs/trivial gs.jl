using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^8
# lsLx = 4:2:12
# for Lx in lsLx
Lx = 4
Ly = 1
params = (J=1,)

Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

J = 1
Δ = 0.5
H = 0

# JE = let LocalSpace = TrivialSpinOneHalf, Root = InteractionTreeNode()
#     je = -1im*J^2*Δ/2
#     for i in 1:size(Latt)-2
#         addIntr!(Root,(LocalSpace.S₊, LocalSpace.Sz, LocalSpace.S₋),(i,i+1,i+2),("S₊","Sz","S₋"),(false,false,false),je,nothing)
#         addIntr!(Root,(LocalSpace.S₋, LocalSpace.Sz, LocalSpace.S₊),(i,i+1,i+2),("S₋","Sz","S₊"),(false,false,false),-je,nothing)
#         addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₊, LocalSpace.S₋),(i,i+1,i+2),("Sz","S₊","S₋"),(false,false,false),-je,nothing)
#         addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₋, LocalSpace.S₊),(i,i+1,i+2),("Sz","S₋","S₊"),(false,false,false),je,nothing)
#         addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋, LocalSpace.Sz),(i,i+1,i+2),("S₊","S₋","Sz"),(false,false,false),je,nothing)
#         addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊, LocalSpace.Sz),(i,i+1,i+2),("S₋","S₊","Sz"),(false,false,false),-je,nothing)
#     end
#     # AutomataSparseMPO(Root,size(Latt))
#     Root
# end

# root = CompositeObservableTreeNode((JE,deepcopy(JE)))
# buildtree!(root)
# treesize(root)



H = TrivialHamiltonian(Latt;params...)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12))
# @save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
# @save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# # end

lsEg

