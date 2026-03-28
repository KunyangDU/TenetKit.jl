using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma/data/ZZHC"

D = 200
Lx = 4
Ly = 2

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)

Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

# Hf = 0.

θ = pi / 2
ϕ = pi / 2

K = 1
Γ = 0.0

for Hf in 0:0.04:0.8


params1_Kitaev = (J1 = J, K1 = K, Γ1 = Γ, Γ1′ = Γ1′)
params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

params_H = (Hx = Hx, Hy = Hy, Hz = Hz)
@show params_H

params = merge(params1,params23,params_H)
params_Kitaev = merge(params1_Kitaev,params23,params_H)

println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsinfo
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ

@time "calculate observables" begin
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

    calObs!(Obs, ψ)
end

obsdata = Obs.values

gsdata = Dict(
    "obs" => obsdata,
    "SE" => lsinfo[end].S
)

@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata


end