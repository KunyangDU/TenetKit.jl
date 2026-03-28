using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 30
Lx = 10
Ly = 1
params = (J=1, Δ = 0.5)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

J = params.J
Δ = params.Δ

H = let LocalSpace = TrivialSpinOneHalf, Root = InteractionTreeNode()
    for i in 3:size(Latt)-1
        addIntr!(Root,LocalSpace.SxSx,(i,i+1),("Sx","Sx"),(false,false),J*Δ,nothing)
        addIntr!(Root,LocalSpace.SySy,(i,i+1),("Sy","Sy"),(false,false),J*Δ,nothing)
        addIntr!(Root,LocalSpace.SzSz,(i,i+1),("Sz","Sz"),(false,false),J,nothing)
    end
    AutomataSparseMPO(Root,size(Latt)) 
end

H3 = let LocalSpace = TrivialSpinOneHalf, Root = InteractionTreeNode()   
    id = isometry(LocalSpace.PhySpace,LocalSpace.PhySpace) 
    for i in 3:size(Latt)-1
        addIntr!(Root,(id,LocalSpace.Sx,LocalSpace.Sx),(2,i,i+1),("I","Sx","Sx"),(false,false,false),J*Δ,nothing)
        addIntr!(Root,(id,LocalSpace.Sy,LocalSpace.Sy),(2,i,i+1),("I","Sy","Sy"),(false,false,false),J*Δ,nothing)
        addIntr!(Root,(id,LocalSpace.Sz,LocalSpace.Sz),(2,i,i+1),("I","Sz","Sz"),(false,false,false),J,nothing)
    end
    AutomataSparseMPO(Root,size(Latt)) 
end

EnvH = Environment([ψ,H,ψ'])
initialize!(EnvH)
@show _scalar(EnvH)
EnvH3 = Environment([ψ,H3,ψ'])
initialize!(EnvH3)
@show _scalar(EnvH3)


lsEg
