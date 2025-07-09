using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/BCAO/arxiv2025/data/Hx"

D = 2^8
Lx =6
Ly = 4

params1_Kitaev = (J1 = -0.59, K1 = -1., Γ1 = 0.53, Γ1′ = 0.11)
params23 = (J2 = -0.038, J3xy = 0.31, J3z = 0.0092)
paramsh = (pinh=0.,)

# params1_Kitaev = (J1 = -0.63, K1 = -1.0, Γ1 = 0.0, Γ1′ = 0.0)
# params23 = (J2 = 0., J3xy = 0.3, J3z = 0.0)
# paramsh = (pinh=0.,)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsh)
params_Kitaev = merge(params1_Kitaev,params23,paramsh)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ

@time "calculate observables" begin
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

    calObs!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
# end
