using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Kitaev/data"

D = 2^7
Lx = 4
Ly = 4

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
xbonds,ybonds,zbonds = getxyzbonds(Latt)

Obs = let 
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf

    for pair in xbonds
        addObs!(Obs,LocalSpace.SxSx,pair,("Sx","Sx"),nothing)
    end
    for pair in ybonds
        addObs!(Obs,LocalSpace.SySy,pair,("Sy","Sy"),nothing)
    end
    for pair in zbonds
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
    end

    Obs
end 

Np = 25
# @load "$(dataname)/params_$(Np).jld2" params
params = [[0.5,0.5,Jz] for Jz in 0:0.1:2]
for (i,(Jx,Jy,Jz)) in enumerate(params)
    @show i/length(params)
    tmpparams = (Jx=Jx,Jy=Jy,Jz=Jz)

    @load "$(dataname)/sweep/lsEg_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" lsEg
    @load "$(dataname)/sweep/ψ_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" ψ

    tmpObs = deepcopy(Obs)
    calObs!(tmpObs, ψ)
    gsdata = tmpObs.values
    @save "$(dataname)/sweep/gsdata_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" gsdata
end
