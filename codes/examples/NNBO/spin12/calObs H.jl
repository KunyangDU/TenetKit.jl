using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
include("../geometry.jl")
dataname = "examples/NNBO/spin12/data/H"

D = 2^6
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

Obs = Observable()
LocalSpace = TrivialSpinOneHalf

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    addObs!(Obs,LocalSpace.Sy,i,"Sy",nothing)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
end

for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),nothing)
    addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),nothing)
    addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
end

for H in 0.2:0.4:2.8
    params1_1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
    params1_3_JDH = (J3 = 1, D = 3)
    paramsh = (h=0.0, H = H)

    params1_cry = _Cub2Cry(params1_1_Kitaev)
    params = merge(params1_cry,params1_3_JDH,paramsh)
    params_Kitaev = merge(params1_1_Kitaev,params1_3_JDH,paramsh)

    @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
    @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ

    calObs!(Obs, ψ;destroy = false)
    gsdata = Obs.values

    @save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
end
