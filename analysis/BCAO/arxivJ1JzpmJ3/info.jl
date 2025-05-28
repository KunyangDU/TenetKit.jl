using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/BCAO/arxivJ1JzpmJ3/data/pin"
figurename = "BCAO/arxivJ1JzpmJ3/figures"
D = 2^6
Lx = 4
Ly = 4
params = (J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
lsinfo

