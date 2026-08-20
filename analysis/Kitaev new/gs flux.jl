using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1"

Lx = 8
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 128
lsHc = 0.01:0.01:0.1
K = -1.0
direction = [[1,0],[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]

fluxsites,fluxdirections,_ = getPBCflux(Latt,XCTria(Lx-1,Ly),direction;edge_shift = [0,sqrt(3)/2],flux_shift = [1,sqrt(3)/3],d = sqrt(3)/3)
lsflux = zeros(length(fluxsites),length(lsHc))
lsS = zeros(3,size(Latt),length(lsHc))

for (i,Hc) in enumerate(lsHc)
params = (K = K, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

edgepoints = vcat(1:4Ly + 2, size(Latt)-4Ly - 1:size(Latt))


Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

for ifl in eachindex(fluxsites)
    dir = collect(fluxdirections[ifl])
    lsflux[ifl,i] = real(gsdata[Snames[dir]][fluxsites[ifl]])
end

for iL in 1:size(Latt)
    lsS[:,iL,i] = map(x -> real(obs[(x,)][(i,)]), collect(Snames))
end

end

figsize = (width = 300,height = 150)

fig = Figure()

ax = Axis(fig[1,1]; 
figsize...,
ylabel = L"W_p",
xlabel = L"B",
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\nK = $(K)", 
)

# plotLatt!(ax,Latt;site = true, sitelabel = false,bond = false)
scatterlines!(ax,lsHc,mean(lsflux,dims = 1)[:] * 2^6)
xlims!(ax,extrema(lsHc))
ylims!(0,1)
resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_Hc_flux_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/Nernst_Hc_flux_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


# end