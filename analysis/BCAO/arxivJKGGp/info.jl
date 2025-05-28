using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/BCAO/arxivJKGGp/data"
figurename = "BCAO/arxivJKGGp/figures"
D = 2^6
Lx =4
Ly = 4
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=0.0,J3xy=0.0,J3z=0.0)
# params = (J1=-0.1,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.038,J3xy=0.3,J3z=0.01)
params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.008)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
lsinfo

