using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")


Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
#@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
D = 2^7

tailname = ""
dataname = "examples/TrivialSpinlessFermion/data"

H = Hamiltonian(Latt;params...)

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ
# @load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsρ
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsF
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsE
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ2
@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" data

Ce = data["Ce"]

lsF .- fe.(lsβ2,Lx,Ly) * L
(lsE .- ue.(lsβ2,Lx,Ly) * L) ./ lsE
# Ceerr = Ce .- ce.(lsβ2,Lx,Ly) * L

