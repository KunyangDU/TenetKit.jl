
using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/U1"

D = 2^7
Lx = 6
Ly = 6
params = (Jz=1,Jxy=0.5,h=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

to = TimerOutput()

@timeit to "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = U₁Spin

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    # for pair in neighbor(Latt)
    #     addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
    # end

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

to

