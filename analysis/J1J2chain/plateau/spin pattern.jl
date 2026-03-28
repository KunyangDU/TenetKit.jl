using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../../analysis/analysis.jl")
include("model.jl")
Ly = 1
Lx = 41
dataname = "../codes/examples/J1J2chain/plateau/data"
@load "$dataname/Latt_$(Lx)x$(Ly).jld2" Latt
D = 128
H = 0.2
params = (J1 = -1, J2 = 0.5, J1xy = 0.0, Hx = 0.0, Hy = 0.0, Hz = H)
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

Sz = [gsdata[(("Sz",),)][((i,),)] for i in 1:size(Latt)]


fig = Figure()
ax = Axis(fig[1,1])

scatter!(ax,1:size(Latt),Sz)

resize_to_layout!(fig)
display(fig)

sum(Sz)/size(Latt)