using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/triangle/trivial"

D = 80
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

Obs = MPSObservable()
LocalSpace = TrivialSpinOneHalf

for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),nothing)
    addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),nothing)
    addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
end 

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    addObs!(Obs,LocalSpace.Sy,i,"Sy",nothing)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
end

lsH = 0:0.2:5
for H in lsH
    @show H
    params = (J=1,H=H)

    @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
    tmpObs = deepcopy(Obs)
    calObs!(tmpObs, ψ)
    gsdata = tmpObs.values

    @save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
end
