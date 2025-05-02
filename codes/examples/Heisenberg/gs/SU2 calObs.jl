using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/SU2"

D = 2^8
Lx = 4
Ly = 4
params = (J=1,)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = SU₂Spin

    for pair in neighbor(Latt)
        addObs!(Obs,LocalSpace.SS,pair,("S","S"),nothing)
    end

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
