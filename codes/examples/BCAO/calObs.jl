using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/BCAO/data"

D = 2^7
Lx = 4
Ly = 2
# params = (J1xy = -1, J1z = -0.36, J2 = 0.023, J3xy = 0.26, J3z = 0.0078,Jpm = 0.023,Jzpm = -0.57)
params = (J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
xbonds,ybonds,zbonds = getxyzbonds(Latt)

@time "calculate observables" begin
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

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

