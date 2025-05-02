using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Heisenberg/data/triangle"

D = 2^7
Lx = 6
Ly = 6
params = (Jz=1,Jxy=0.5,h=0)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg


SSxy = zeros(size(Latt),size(Latt))
SSzz = zeros(size(Latt),size(Latt))

for i in 1:size(Latt), j in i+1:size(Latt)
    SSzz[i,j] = gsdata["SzSz"][(i,j)]
    SSxy[i,j] = gsdata["S+S-"][(i,j)] + gsdata["S-S+"][(i,j)]
end
SSzz .+= SSzz'
SSxy .+= SSxy'
SSzz = SSzz .+ diagm(ones(size(Latt))) / 4
SSxy = SSxy / 2  .+ diagm(ones(size(Latt))) / 2


lskx = 2*0.999*pi*range(-1,1,51)
lsky = 4pi/sqrt(3)*range(-1,1,51)
lsk = filter(x -> isinside(x,MFBZpoint;isboundary = true),[[kx,ky] for kx in lskx,ky in lsky][:])
lstk = map(x -> Tuple(x),lsk)
FSSzz = FT2(SSzz,Latt,lstk)
FSSxy = FT2(SSxy,Latt,lstk)
Sm = max(maximum(FSSzz),maximum(FSSxy))

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

Sm = maximum((FSSxy .+ FSSzz))

hmt = heatmap!(axt,x,y,FSSzz .+ FSSxy,colorrange = (0., Sm))
hmxy = heatmap!(axxy,x,y,FSSxy,colorrange = (0., Sm))
hmzz = heatmap!(axzz,x,y,FSSzz,colorrange = (0., Sm))

L"-\frac{2\pi}{3}",L"0",L"\frac{2\pi}{3}",L"\frac{4\pi}{3}"

for ax in [axt,axxy,axzz]
    boundary!(ax,FBZpoint;color = :white,linewidth = 1.5*ax.height.val/250,linestyle = :dash)
    boundary!(ax,MFBZpoint;linewidth = 5*ax.height.val/250,breathing=1.015)
    ylims!(ax,-4pi/sqrt(3)-0.3,4pi/sqrt(3)+0.3)
    xlims!(ax,-2pi,2pi)
end

hidexdecorations!(axxy,grid = false,ticks = false)


Colorbar(fig[1:2,3],hmt,
# label = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle",
)


Label(fig[1,1:3][1, 1, Top()], "Structure factor, $(Ly)x$(Lx) XC-Tria-CY, D = $(D)",
fontsize = 15,
font = :bold,
padding = (0, 0, 25, 0),
halign = :center)

resize_to_layout!(fig)
display(fig)

save("triangle/figures/structure factor_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("triangle/figures/structure factor_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)





