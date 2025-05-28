using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

figurename = "BCAO/arxivJKGGp/figures"
dataname = "../codes/examples/BCAO/arxivJKGGp/data/pin"
pinsites = vcat(1:2Ly,size(Latt)-2Ly+1:size(Latt))
# dataname = "../codes/examples/BCAO/arxivJKGGp/data"
# pinsites = []
D = 2^6
Lx = 6
Ly = 4
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=0.0,J3xy=0.0,J3z=0.0)
# params = (J1=-0.1,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.46,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.008)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.28,J3z=0.0078)
# params = (J1=-0.50035,K1=-0.85893,Γ1=0.4538,Γ1′=0.09311,J2= -0.0321,J3xy=0.26,J3z=0.0078)

# 6x4
params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.28,J3z=0.0078)

# non pin
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.008)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.038,J3xy=0.3,J3z=0.01)

# pin
# params = (J1=-0.516,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.036,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.038,J3xy=0.26,J3z=0.0078)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg

lskx = 4pi/sqrt(3)*range(-1,1,51)
lsky = 2*0.999*pi*range(-1,1,51)
lsk = filter(x -> isinside(x,MFBZpoint;isboundary = true),[[kx,ky] for kx in lskx,ky in lsky][:])
lstk = map(x -> Tuple(x),lsk)

SxSx = zeros(size(Latt),size(Latt))
SySy = zeros(size(Latt),size(Latt))
SzSz = zeros(size(Latt),size(Latt))
for i in 1:size(Latt),j in i+1:size(Latt)
    i in pinsites && continue
    j in pinsites && continue
    SxSx[i,j] = gsdata["SxSx"][(i,j)]
    SySy[i,j] = gsdata["SySy"][(i,j)]
    SzSz[i,j] = gsdata["SzSz"][(i,j)]
end
FSxSx,FSySy,FSzSz = map(x -> (1- length(pinsites)/size(Latt)) * FT2(x .+= x' .+ diagm(ones(size(Latt)))/4,Latt,lstk),[SxSx,SySy,SzSz])

x = map(lsk) do k
    k[1]
end
y = map(lsk) do k
    k[2]
end

Sm = maximum(FSxSx .+ FSySy .+ FSzSz)
Smd = maximum(vcat(FSxSx, FSySy, FSzSz))
figsize = (width = 180,height = 180*sqrt(3)/2)

fig = Figure()
axt = Axis(fig[1,1];autolimitaspect = true,figsize...,
xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
title = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle")
axx = Axis(fig[1,2];autolimitaspect = true,figsize...,
xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
title = L"\langle S_x\cdot S_x\rangle")
axy = Axis(fig[2,1];autolimitaspect = true,figsize...,
xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
title = L"\langle S_y\cdot S_y\rangle")
axz = Axis(fig[2,2];autolimitaspect = true,figsize...,
xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
title = L"\langle S_z\cdot S_z\rangle")
hmt = heatmap!(axt,x,y,FSxSx .+ FSySy .+ FSzSz,colorrange = (0,Sm);colormap = :Reds)
hmx = heatmap!(axx,x,y,FSxSx,colorrange = (0,Smd);colormap = :Reds)
hmy = heatmap!(axy,x,y,FSySy,colorrange = (0,Smd);colormap = :Reds)
hmz = heatmap!(axz,x,y,FSzSz,colorrange = (0,Smd);colormap = :Reds)

# L"-\frac{2\pi}{3}",L"0",L"\frac{2\pi}{3}",L"\frac{4\pi}{3}"

for ax in [axt,axx,axy,axz]
    boundary!(ax,FBZpoint;color = :black,linewidth = 1.2,linestyle = :dash)
    boundary!(ax,dZZFBZpoint;color = :black,linewidth = 1.2,linestyle = :dash)
    boundary!(ax,MFBZpoint;linewidth = 5,breathing=1.015)
    xlims!(ax,-4pi/sqrt(3),4pi/sqrt(3))
    ylims!(ax,-2pi-0.3,2pi+0.3)
end

hidexdecorations!(axt,grid = false,ticks = false)
hidexdecorations!(axx,grid = false,ticks = false)
hideydecorations!(axx,grid = false,ticks = false)
hideydecorations!(axz,grid = false,ticks = false)



Colorbar(fig[1,3],hmt,label = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle")
Colorbar(fig[2,3],hmx,label = L"\langle S_\alpha S_\alpha\rangle")

Label(fig[1,1:2][1, 1, Top()], "Structure factor, $(Ly)x$(Lx)x2 ZZ-HC-CY, D = $(D), J₁-J₃",
fontsize = 15,
font = :bold,
padding = (0, 0, 25, 0),
halign = :center)

resize_to_layout!(fig)
display(fig)

save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

