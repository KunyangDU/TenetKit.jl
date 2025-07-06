using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/NNBO/data"

D = 3^4
Lx = 4
Ly = 4

params1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
params3DH = (J3 = 1, D = -3,  H = 0.)
paramsh = (h=0.01,)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params3DH,paramsh)
params_Kitaev = merge(params1_Kitaev,params3DH,paramsh)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ

@time "calculate observables" begin
    Obs = Observable()
    LocalSpace = TrivialSpinOne

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
    # _calObs_threading!(Obs, ψ)
end

gsdata = Obs.values

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
# end


