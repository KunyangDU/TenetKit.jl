using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Kitaev/data"

D = 2^7
Lx = 4
Ly = 2
params = (Jx=0.333,Jy=0.333,Jz=0.333)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/sweep/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/sweep/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
xbonds,ybonds,zbonds = getxyzbonds(Latt)

@time "calculate observables" begin
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

    # for i in 1:size(Latt)
    #     addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    #     addObs!(Obs,LocalSpace.Sy,i,"Sy",nothing)
    #     addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    # end

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/sweep/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

