
using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("model.jl")

trivialname = "../codes/examples/Heisenberg/data/triangle/trivial"
figurename = "tanTRG/structure factor"

D = 2^4
Lx = 4
Ly = 4
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
trivialparams = (J=1,)

@load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(trivialparams).jld2" lsβ2
@load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(trivialparams).jld2" data

for (iβ,β) in enumerate(lsβ2)
    obsdata = data["obs"][iβ]

    lskx = 2*0.999*pi*range(-1,1,51)
    lsky = 4pi/sqrt(3)*range(-1,1,51)
    lsk = filter(x -> isinside(x,MFBZpoint;isboundary = true),[[kx,ky] for kx in lskx,ky in lsky][:])
    lstk = map(x -> Tuple(x),lsk)
    FSxSx,FSySy,FSzSz = TrivialSSFT(Latt,obsdata,lstk,Ly+1:size(Latt)-Ly)

    Sm = max(1/4, maximum(FSzSz .+ FSxSx .+ FSySy))

    x = map(lsk) do k
        k[1]
    end
    y = map(lsk) do k
        k[2]
    end

    figsize = (width = 105*sqrt(3)/2,height = 105)
    figsizet = (width = 250*sqrt(3)/2,height = 250)

    fig = Figure()
    axt = Axis(fig[1:2,2];autolimitaspect = true,figsizet...,
    xlabel = L"k_x\ /\ 2\pi",ylabel = L"k_y\ /\ \left(2\pi\sqrt{3}/3\right)",
    xticks = (2pi*(-1:0.5:1),string.(-2:2)),yticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:1:2)),
    title = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle")
    axxy = Axis(fig[1,1];autolimitaspect = true,figsize...,
    xlabel = L"k_x\ /\ 2\pi",ylabel = L"k_y\ /\ \left(2\pi\sqrt{3}/3\right)",
    xticks = (2pi*(-1:0.5:1),string.(-2:2)),yticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:1:2)),
    title = L"\langle S_xS_x + S_yS_y\rangle")
    axzz = Axis(fig[2,1];autolimitaspect = true,figsize...,
    xlabel = L"k_x\ /\ 2\pi",ylabel = L"k_y\ /\ \left(2\pi\sqrt{3}/3\right)",
    xticks = (2pi*(-1:0.5:1),string.(-2:2)),yticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:1:2)),
    title = L"\langle S_z S_z\rangle")


    hmt = heatmap!(axt,x,y,FSxSx .+ FSySy .+ FSzSz,colorrange = (0., Sm))
    hmxy = heatmap!(axxy,x,y,FSxSx .+ FSySy,colorrange = (0., Sm))
    hmzz = heatmap!(axzz,x,y,FSzSz,colorrange = (0., Sm))

    for ax in [axt,axxy,axzz]
        boundary!(ax,FBZpoint;color = :white,linewidth = 1.5*ax.height.val/250,linestyle = :dash)
        boundary!(ax,MFBZpoint;linewidth = 5*ax.height.val/250,breathing=1.015)
        ylims!(ax,-4pi/sqrt(3)-0.3,4pi/sqrt(3)+0.3)
        xlims!(ax,-2pi,2pi)
    end

    hidexdecorations!(axxy,grid = false,ticks = false)

    Colorbar(fig[1:2,3],hmt)

    Label(fig[1,1:3][1, 1, Top()], "Structure factor, $(Ly)x$(Lx) XC-Tria-CY, D = $(D), T = $(round(1/β;digits = 3)) J",
    fontsize = 15,
    font = :bold,
    padding = (0, 0, 25, 0),
    halign = :center)

    resize_to_layout!(fig)
    display(fig)

    save("Heisenberg/triangle/figures/$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
    save("Heisenberg/triangle/figures/$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
end

