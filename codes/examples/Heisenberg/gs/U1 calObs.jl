
using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/U1"

D = 2^6
Lx = 10
Ly = 1
params = (Jz = 1,Jxy = 0.5,H=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

to = TimerOutput()

@timeit to "calculate observables" begin
    Obs = Observable()
    LocalSpace = U₁Spin

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for i in 1:size(Latt), j in i+1:size(Latt)
        pair = (i,j)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
        addObs!(Obs,LocalSpace.S₊S₋,pair,("S+","S-"),nothing)
        addObs!(Obs,LocalSpace.S₋S₊,pair,("S-","S+"),nothing)
    end

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

