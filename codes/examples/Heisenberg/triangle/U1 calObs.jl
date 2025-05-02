
using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/triangle"

D = 2^7
Lx = 6
Ly = 6
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

lsH = 0:0.2:5
for H in lsH
    @show H
    params = (Jz = 1,Jxy = 0.5,H=H)
    # @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
    @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

    @time "calculate observables" begin
        Obs = MPSObservable()
        LocalSpace = U₁Spin

        for i in 1:size(Latt)
            addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
        end

        for i in 1:size(Latt),j in i+1:size(Latt)
            addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
            addObs!(Obs,LocalSpace.S₊S₋,(i,j),("S+","S-"),nothing)
            addObs!(Obs,LocalSpace.S₋S₊,(i,j),("S-","S+"),nothing)
        end

        calObs!(Obs, ψ)
    end

    gsdata = Obs.values

    @save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

end

