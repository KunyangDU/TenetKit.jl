using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data"

D = 2^7
Lx = 4
Ly = 2
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (Jx=0.333,Jy=0.333,Jz=0.333)

xbonds,ybonds,zbonds = getxyzbonds(Latt)
# selectedpoints = 17:48
# selectedpoints = 1:size(Latt)
selectedpoints = 9:24

@load "$(dataname)/sweep/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@load "$(dataname)/sweep/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @show (Jx,Jy,Jz),lsEg[end]
ssz = [gsdata["SzSz"][pair] for pair in filter(x -> x[1] in selectedpoints && x[2] in selectedpoints,zbonds)] |> x -> sum(x)/length(x)
lskx = 2pi/sqrt(3)*range(-1,1,21)
lsky = 4pi/3*range(-1,1,21)
lsk = filter(x -> isinside(x,FBZpoint),[[kx,ky] for kx in lskx,ky in lsky][:])
theossz = getKitaevSS(lsk,[params.Jx,params.Jy,params.Jz])

(ssz - theossz)/theossz
