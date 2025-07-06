using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^6
Lx = 10
Ly = 1
params = (J=1, H = 1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

@time "calculate observables" begin
    Obs = Observable()
    LocalSpace = TrivialSpinOneHalf

    for i in 1:size(Latt), j in i+1:size(Latt)
        pair = (i,j)
        addObs!(Obs,LocalSpace.SxSx,pair,("Sx","Sx"),nothing)
        addObs!(Obs,LocalSpace.SySy,pair,("Sy","Sy"),nothing)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
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

