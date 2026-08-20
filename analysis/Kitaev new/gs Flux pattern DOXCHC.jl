using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/DOXCHC"

Lx = 12
Ly = 1
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 200
Hc = 0.1
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction = [[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]]
# edgepoints = vcat(1:Ly + 2, size(Latt)-Ly - 1:size(Latt))
edgepoints = []

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

flux_Latt = XCTria(Lx,4)

fluxsites,fluxdirections,_ = getOBCflux(Latt,XCTria(Lx,4),direction;total_shift = [-3/2,sqrt(3)/6],d = sqrt(3)/3)

wp = merge(map(x -> obs[x],filter(x -> length(x) == 6,collect(keys(obs))))...)
effsites,wpvals = let effsites = Int64[],wpvals = Float64[]
    for (i,sites) in enumerate(fluxsites)
        if haskey(wp,sites)
            push!(effsites,i)
            push!(wpvals,real(wp[sites]))
        end
    end
    effsites,wpvals
end



figsize = (width = 80*Lx,height = 160*Ly*sqrt(3))

fig = Figure()

ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (6.5 .+ 1.0 .* (0:Lx-1),string.(1:Lx)),
yticks = (sqrt(3)/6 + sqrt(3) .+ sqrt(3) * (0:Ly-1),string.(1:Ly)),
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
# xgridvisible = false,
# ygridvisible =false,
)

polyHexagon!(ax,map(x -> collect(coordinate(flux_Latt,x)),effsites),get(colorschemes[:bwr], wpvals * 2^6,extrema( wpvals * 2^6));scale = 1/sqrt(3),total_shift = [-3/2,sqrt(3)/6])
plotLatt!(ax,Latt;site = true,tplevel=1, bond = false)

Colorbar(fig[1,2],colormap = :bwr,colorrange = extrema( wpvals * 2^6))

hidedecorations!(ax)

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Flux_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/Flux_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
