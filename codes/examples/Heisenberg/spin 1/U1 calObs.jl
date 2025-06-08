
using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Heisenberg/spin 1/data/U1"

D = 3^4
Lx = 8
Ly = 1
params = (Jz=1,Jxy=1/2,D=0)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

to = TimerOutput()

@timeit to "calculate observables" begin
    Obs = Observable()
    LocalSpace = U₁Spin1

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
        addObs!(Obs,LocalSpace.Sz2,i,"Sz2",nothing)
    end

    for pair in neighbor(Latt)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
    end

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata



