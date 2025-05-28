using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

trivialname = "../codes/examples/Heisenberg/data/triangle/pin"
figurename = "tanTRG/structure factor"
typename = "pin"
D = 2^9
Lx = 6
Ly = 4
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J=1,)

@load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
@load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

# for (iβ,β) in enumerate(lsβ2)
#     obsdata = data["obs"][iβ]
#     Sx = [obsdata["Sx"][(i,)] for i in 1:size(Latt)]
#     Sy = [obsdata["Sy"][(i,)] for i in 1:size(Latt)]
#     Sz = [obsdata["Sz"][(i,)] for i in 1:size(Latt)]
#     @show Sz

#     Sm = maximum(sqrt.(Sx .^ 2 + Sy .^ 2))

#     figsize = (width = 60*Lx,height = 60*(Ly)*sqrt(3)/2)

#     fig = Figure()
#     ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
#     xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)),
#     title = "$(Lx)x$(Ly) ZZ-HC-CY, D=$(D)")
#     plotLatt!(ax,Latt,[0,sqrt(3)/2];site = false,sitelabel = false,sitesize = 12*ones(size(Latt)))

#     intensity = 0.7 / (sqrt.(Sx .^ 2 + Sy .^ 2) |> x -> sum(x)/length(x))
#     arrcolor = [:Reds,:Blues,:Greens]
#     arrowcolors(x) = get(colorschemes[arrcolor[x[1]]],x[2],(0.,1/2))
#     colors = map(x -> arrowcolors(orientate3(x)),collect.(eachcol(hcat(Sx,Sy)')))
#     for i in 1:size(Latt)
#         arrowc!(ax,coordinate(Latt,i)...,intensity*Sx[i],intensity*Sy[i];color = colors[i],linewidth = 2)
#     end

#     Colorbar(fig[1,2],limits = (0,1/2),colormap = :Reds,width = 8,ticksvisible = true,ticklabelsvisible = false)
#     Colorbar(fig[1,3],limits = (0,1/2),colormap = :Blues,width = 8,ticksvisible = true,ticklabelsvisible = false)
#     Colorbar(fig[1,4],limits = (0,1/2),colormap = :Greens,width = 8,label = L"\langle \mathbf{S} \cdot \mathbf{\hat{e}}\rangle")

#     # axx = Axis(fig[1,1];autolimitaspect = true,figsize...)
#     # axy = Axis(fig[2,1];autolimitaspect = true,figsize...)
#     # axz = Axis(fig[3,1];autolimitaspect = true,figsize...)

#     # plotLatt!(axx,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sx,(-Sm,Sm)))
#     # plotLatt!(axy,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sz,(-Sm,Sm)))
#     # plotLatt!(axz,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sz,(-Sm,Sm)))

#     # Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr)
#     # Colorbar(fig[2,2],limits = (-Sm,Sm),colormap = :bwr)
#     # Colorbar(fig[3,2],limits = (-Sm,Sm),colormap = :bwr)

#     resize_to_layout!(fig)
#     display(fig)
#     save("Heisenberg/triangle/figures/$(typename)/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
#     save("Heisenberg/triangle/figures/$(typename)/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

# end