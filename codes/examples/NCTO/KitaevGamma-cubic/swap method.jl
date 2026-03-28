using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma-cubic/data/OHHC"

D = 20
L = 1

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)

@load "$(dataname)/Latt_$(L).jld2" Latt

# Hf = 0.

θ = 0.0
ϕ = pi / 2

K = 1
Γ = 0.0

params1_Kitaev = (K = K, Γ = Γ)
Hf = 0.0
Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

println("$(size(Latt))), D = $(D), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/lsEg_$(L)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/lsinfo_$(L)_$(D)_$(params_Kitaev).jld2" lsinfo
@load "$(dataname)/ψ_$(L)_$(D)_$(params_Kitaev).jld2" ψ

# vonNeumann(ψ,[1,2,3,4]) 
area = Dict(
    "A" => [2,3,4,10,11,18],
    "B" => [1,5,6,7,14,15],
    "C" => [19,20,21,22,23,24],
    "D" => [8,9,12,13,16,17]
)

SA = vonNeumann(ψ,area["A"])
SB = vonNeumann(ψ,area["B"])
SC = vonNeumann(ψ,area["C"])
SAB = vonNeumann(ψ,vcat(area["A"],area["B"]))
SBC = vonNeumann(ψ,vcat(area["B"],area["C"]))
SAC = vonNeumann(ψ,vcat(area["A"],area["C"]))
SABC = vonNeumann(ψ,vcat(area["A"],area["B"],area["C"]))

TEE = (SA + SB + SC - SAB - SBC - SAC + SABC) / log(2)

# sort(vcat(values(area)...)) == collect(1:size(Latt))



