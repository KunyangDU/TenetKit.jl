using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/BCAO/data/yeesuan"

D = 2^7
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

lsHx = vcat(0:0.02:0.2,0.25:0.05:1)
for Hx in lsHx
    # params = (J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
    params = (Hx = Hx, J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)

    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

    Sx = [gsdata["Sx"][(i,)] for i in 1:size(Latt)]
    Sy = [gsdata["Sy"][(i,)] for i in 1:size(Latt)]
    Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]

    Sm = maximum(abs.(vcat(Sx,Sy)))

    figsize = (width = Lx*sqrt(3)*50,height = Ly*60)

    fig = Figure()
    ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
    xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
    yticks = (1:0.5:Ly+0.5,string.(1:2Ly)),
    title = "$(Ly)x$(Lx)x2 ZZ-HC-CY, D = $(D), μ₀Hₓ = $(round(Hx*6.54;digits = 2)) T")
    plotLatt!(ax,Latt,[0,1];site = true,sitelabel = false)


    intensity = 0.3 / maximum(sum(sqrt.(Sx.^2 .+ Sy.^2))/length(Sx))
    colors = Hx == 0 ? get(colorschemes[:bwr],Sy,(-Sm/3,Sm/3)) : get(colorschemes[:bwr],Sx,(-Sm/3,Sm/3))
    for i in 1:size(Latt)
        arrowc!(ax,coordinate(Latt,i)...,intensity*Sx[i],intensity*Sy[i];color = colors[i],linewidth = 2)
    end

    label = Hx == 0 ? L"\langle S_y\rangle" : L"\langle S_x \rangle"

    Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr,label=label)

    # axx = Axis(fig[1,1];autolimitaspect = true,figsize...)
    # axy = Axis(fig[2,1];autolimitaspect = true,figsize...)
    # axz = Axis(fig[3,1];autolimitaspect = true,figsize...)

    # plotLatt!(axx,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sx,(-Sm,Sm)))
    # plotLatt!(axy,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sy,(-Sm,Sm)))
    # plotLatt!(axz,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sz,(-Sm,Sm)))

    # Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr)
    # Colorbar(fig[2,2],limits = (-Sm,Sm),colormap = :bwr)
    # Colorbar(fig[3,2],limits = (-Sm,Sm),colormap = :bwr)

    resize_to_layout!(fig)
    display(fig)
    save("BCAO/figures/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
    save("BCAO/figures/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
end