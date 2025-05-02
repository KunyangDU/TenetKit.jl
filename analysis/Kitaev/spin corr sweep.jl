using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data"

D = 2^9
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

Np = 25
# @load "$(dataname)/params_$(Np).jld2" params
params = [[0.5,0.5,Jz] for Jz in 0:0.1:2]
xbonds,ybonds,zbonds = getxyzbonds(Latt)
selectedpoints = 17:48
# selectedpoints = 1:size(Latt)
ssz = zeros(length(params))
for (i,(Jx,Jy,Jz)) in enumerate(params)
    tmpparams = (Jx=Jx,Jy=Jy,Jz=Jz)
    @load "$(dataname)/yeesuan/gsdata_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" gsdata
    # @load "$(dataname)/yeesuan/lsEg_$(Lx)x$(Ly)_$(D)_$(tmpparams).jld2" lsEg
    # @show (Jx,Jy,Jz),lsEg[end]
    ssz[i] = [gsdata["SzSz"][pair] for pair in filter(x -> x[1] in selectedpoints && x[2] in selectedpoints,zbonds)] |> x -> sum(x)/length(x)
end

@save "$(dataname)/ssz_$(Lx)x$(Ly)_$(D)_$(length(params)).jld2" ssz




