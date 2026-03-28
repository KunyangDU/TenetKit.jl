using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")

RBASIS3 = [[sqrt(3)/2,1/2,0.],[sqrt(3)/2,-1/2,0.],[0.,0.,1.]]
KBASIS2 = kbasis2(RBASIS3)
RBASIS2 = map(x -> x[1:2],RBASIS3[1:2])
KitaevBasis2 = [[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]
FBZpoint = [[1/3,2/3],[2/3,1/3],[1/3,-1/3],[-1/3,-2/3],[-2/3,-1/3],[-1/3,1/3]]
BASIS2 = [[1.,0.],[0.,1.]]
Triapoint = [[-sqrt(3)/2,-1/2],[sqrt(3)/2,-1/2],[0.,1]]
JBASIS2 = Triapoint
KitaevBDpoiont = let 
    a,b,c = JBASIS2
    [(a+b)/2,(b+c)/2,(a+c)/2]
end
MFBZpoint = [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]]
dZZFBZpoint = map(x -> x/2,FBZpoint)

dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC/SSF"
tailname = ""

D = 256
Lx = 6
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)


θ = 0.0 * pi
ϕ = 0.5 * pi


K = 1.0
Γ′ = 0.2

params1_Kitaev = (K = K, Γ′ = Γ′)
# params1_Kitaev = (K = K, Γ = Γ′)

for Hf in 1.04
# Hf = 0.0

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)
Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
params = merge(params1_Kitaev,params_H)
params_Kitaev = merge(params1_Kitaev,params_H)
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
gsdata = gsdata["obs"]

lsky = 2*pi*0.999*range(-1,1,125)
lskx = pi*4sqrt(3)/3*range(-1,1,125)
lsk = filter(x -> isinside(x,
MFBZpoint
;isboundary = true,
# basis = kbasis2(rbasis223(Latt.subLatts[1].e))
),[[kx,ky] for kx in lskx,ky in lsky][:])
lstk = map(x -> Tuple(x),lsk)

# SxSx1 = getCorrMat(Latt,gsdata["SxSx"],1/4)

# SxSx = zeros(size(Latt),size(Latt))
# SySy = zeros(size(Latt),size(Latt))
# SzSz = zeros(size(Latt),size(Latt))
# for i in 1:size(Latt),j in i+1:size(Latt)
#     SxSx[i,j] = gsdata["SxSx"][(i,j)]
#     SySy[i,j] = gsdata["SySy"][(i,j)]
#     SzSz[i,j] = gsdata["SzSz"][(i,j)]
# end
FSxSx,FSySy,FSzSz = map(x -> FT2(getCorrMat(Latt,gsdata[(x,)],1/4;selected_point = 1:size(Latt),independent_data = gsdata[(tuple(x[1]),)]),Latt,lstk),[("Sx","Sx"),("Sy","Sy"),("Sz","Sz")])

x = map(lsk) do k
    k[1]
end
y = map(lsk) do k
    k[2]
end

Sm = maximum(FSxSx .+ FSySy .+ FSzSz)
Smd = maximum(vcat(FSxSx,FSySy,FSzSz))*1.
figsize = (width = 200,height = 200*sqrt(3)/2)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
title = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle")
# axx = Axis(fig[1,2];autolimitaspect = true,figsize...,
# xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
# xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
# title = L"\langle S_x\cdot S_x\rangle")
# axy = Axis(fig[2,1];autolimitaspect = true,figsize...,
# xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
# xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
# title = L"\langle S_y\cdot S_y\rangle")
# axz = Axis(fig[2,2];autolimitaspect = true,figsize...,
# xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
# xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
# title = L"\langle S_z\cdot S_z\rangle")
hm = heatmap!(ax,x,y,FSxSx .+ FSySy .+ FSzSz,colormap = :Reds)
# hmx = heatmap!(axx,x,y,FSxSx,colorrange = (0,Smd),colormap = :hsv)
# hmy = heatmap!(axy,x,y,FSySy,colorrange = (0,Smd),colormap = :hsv)
# hmz = heatmap!(axz,x,y,FSzSz,colorrange = (0,Smd),colormap = :hsv)


boundary!(ax,FBZpoint;color = :white,linewidth = 1.5,linestyle = :dash)
boundary!(ax,dZZFBZpoint;color = :red,linewidth = 1.5)
boundary!(ax,MFBZpoint;linewidth = 3,breathing=1.015)

# scatter!(ax,0.27*coordinate(-[1,0];basis = KBASIS2)...;markersize = 8,color = :green,marker = :star4)
# scatter!(ax,0.33*coordinate([1,0];basis = KBASIS2)...;markersize = 8,color = :green,marker = :star4)
# scatter!(ax,0.33*coordinate(-[1,0];basis = KBASIS2)...;markersize = 8,color = :green,marker = :star4)
ylims!(ax,-4pi/sqrt(3),4pi/sqrt(3))
xlims!(ax,-2pi-0.3,2pi+0.3)




Colorbar(fig[1,2],hm,label = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle")

Label(fig[1,1:2][1, 1, Top()], "$(Ly)x$(Lx)x2, D = $(D), $(params1_Kitaev)\n$(params_H)",
fontsize = 15,
font = :bold,
padding = (0, 0, 25, 0),
halign = :center)

resize_to_layout!(fig)
display(fig)

save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).png",fig)
save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).pdf",fig)

end