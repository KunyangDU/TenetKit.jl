using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")

# λ = 0.5
dataname = "examples/J1J2chain/plateau/data"
Ly = 1

Lx = 60
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 128

Obs = Observable()
LocalSpace = TrivialSpinOneHalf

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing)
    addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing)
end

for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),(false,false),nothing)
    addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),(false,false),nothing)
    addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),(false,false),nothing)
end

for H in 0:0.04:0.4
params = (J1 = -1, J2 = 0.5, J1xy = 0.0, Hx = 0.0, Hy = 0.0, Hz = H)
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ


@time "calculate observables" begin
    calObs!(Obs, ψ;destroy = false)
    # _calObs_threading!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
end

