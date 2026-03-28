using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")

dataname = "examples/BCAO/arxiv2025/data"

D = 2^6
Lx = 2
Ly = 2
Ds = 2^4

params1_Kitaev = (J1 = -0.59, K1 = -1, Γ1 = 0.53, Γ1′ = 0.11)
params23 = (J2 = -0.038, J3xy = 0.31, J3z = 0.0092, Hx = 0.)
paramsh = (pinh=0.,)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsh)
params_Kitaev = merge(params1_Kitaev,params23,paramsh)
println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsβ
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsρ
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsE
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsF
lsβ2 = 2 * lsβ[2:end]
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsβ2


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

obs = repeat([Dict(),],length(lsβ2))

Es = lsE
Fs = lsF

for (iβ,β) in enumerate(lsβ2)
    @show iβ / length(lsβ2)
    ρ = lsρ[iβ]
    calObs!(Obs,ρ;destroy = false)
    obs[iβ] = Obs.values
end

data = Dict(
    "E" => lsE,
    "F" => lsF,
    "obs" => obs
)

@save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" data
