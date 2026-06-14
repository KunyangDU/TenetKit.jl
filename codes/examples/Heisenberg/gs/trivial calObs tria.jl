using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/triangular"

D = 128
Lx = 8
Ly = 6
params = (J=1, Δ = 1, hx = 1.0 * sqrt(3)/2, hy = 1.0 * 1/2)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

begin
    Obs = ObservableGraph(size(Latt))
    LocalSpace = TrivialSpinOneHalf

    for i in 1:size(Latt), j in i+1:size(Latt)
        pair = (i,j)
        addObs!(Obs,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),nothing,1.0)
        addObs!(Obs,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),nothing,1.0)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),nothing,1.0)
    end

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing,1.0)
        addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing,1.0)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing,1.0)
    end
    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata



# gsdata[ (("Sz", "Sz"),)][((6,9),)]

gsdata
