using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 128
Lx = 16
Ly = 1
params = (Jxy=0.0, Jz = 1.0, Hx = 0.0, Hy = 0.1, Hz = 0.0)

h = normalize([params.Hx,params.Hy,params.Hz])
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

Obs = let LocalSpace = TrivialSpinOneHalf, Sops = (LocalSpace.Sx,LocalSpace.Sy,LocalSpace.Sz), Snames = ("Sx","Sy","Sz")

    Obs = ObservableGraph(size(Latt))
    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing)
        addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing)
    end
    for i in 1:size(Latt), j in 1:size(Latt)
        i ≥ j && continue
        addObs!(Obs,(LocalSpace.Sx,LocalSpace.Sx),(i,j),("Sx","Sx"),(false,false),nothing)
        addObs!(Obs,(LocalSpace.Sy,LocalSpace.Sy),(i,j),("Sy","Sy"),(false,false),nothing)
        addObs!(Obs,(LocalSpace.Sz,LocalSpace.Sz),(i,j),("Sz","Sz"),(false,false),nothing)
    end

    cinds = currentindex2(diagm([params.Jxy, params.Jxy, params.Jz]),h)
    for (i,j) in neighbor(Latt), (j′,(γ,γ′)) in cinds
        addObs!(Obs,(Sops[γ],Sops[γ′]),(i,j),(Snames[γ],Snames[γ′]),(false,false),nothing)
    end
    
    Obs
end

gsdata = calObs!(Obs,ψ;showtimes = 0)

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

gsdata