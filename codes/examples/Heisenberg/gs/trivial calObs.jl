using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^7
Lx = 6
Ly = 6
params = (J=1,h=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf

    # for pair in neighbor(Latt)
    #     addObs!(Obs,LocalSpace.SxSx,pair,("Sx","Sx"),nothing)
    #     addObs!(Obs,LocalSpace.SySy,pair,("Sy","Sy"),nothing)
    #     addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
    # end

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
        addObs!(Obs,LocalSpace.Sy,i,"Sy",nothing)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

