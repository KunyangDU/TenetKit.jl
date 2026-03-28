using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

RBASIS3 = sqrt(3) .* [[1.,0.,0.],[1/2,sqrt(3)/2,0.],[0.,0.,1.]]
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

dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/OHHC"
figurename = "NCTO/KitaevGamma-cubic/figures/OHHC"
tailname = ""

D = 256
L = 3

@load "$(dataname)/Latt_$(L).jld2" Latt

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)


θ = pi / 2
ϕ = pi / 2



K = 1

lsHf = 0:0.04:0.8
lsΓ = 0:-0.04:-0.4

lsΓ0 = [[0,0],] |> x -> coordinate.(x;basis = KBASIS2)
lsM = [[1,0],[0,1],[1,1],-[1,0],-[0,1],-[1,1]] ./ 2 |> x -> coordinate.(x;basis = KBASIS2)
lsUUD= [[1,0],[0,1],[1,1],-[1,0],-[0,1],-[1,1]] ./ 3 |> x -> coordinate.(x;basis = KBASIS2)
lsΓ2 = [[1,0],[0,1],[1,1],-[1,0],-[0,1],-[1,1]] |> x -> coordinate.(x;basis = KBASIS2)
lsK = [[1,2],[2,1],[1,-1],-[1,2],-[2,1],-[1,-1]]/3 |> x -> coordinate.(x;basis = KBASIS2)
lsK32 = [[1,2],[2,1],[1,-1],-[1,2],-[2,1],-[1,-1]]/2 |> x -> coordinate.(x;basis = KBASIS2)
lstotalpoint = vcat(lsΓ,lsM,lsUUD,lsΓ2,lsK,lsK32)


# lsky = 6*range(-1,1,75)
# lskx = pi*sqrt(3)/2*0.999*pi*range(-1,1,75)
# lsky = 4pi/3*range(-1,1,75)
# lskx = 2*0.999*pi/sqrt(3)*range(-1,1,75)
lsk = lsΓ0
# lsk = [[kx,ky] for kx in lskx,ky in lsky][:]
lstk = map(x -> Tuple(x),lsk)

lsFFTS = zeros(length(lsHf),length(lsΓ))

for (i,Hf) in enumerate(lsHf),(j,Γ) in enumerate(lsΓ)
params1_Kitaev = (K = K, Γ = Γ)
Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)
Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)
@load "$(dataname)/gsdata_$(L)_$(D)_$(params_Kitaev).jld2" gsdata
gsdata = gsdata["obs"]


# SxSx1 = getCorrMat(Latt,gsdata["SxSx"],1/4)

# SxSx = zeros(size(Latt),size(Latt))
# SySy = zeros(size(Latt),size(Latt))
# SzSz = zeros(size(Latt),size(Latt))
# for i in 1:size(Latt),j in i+1:size(Latt)
#     SxSx[i,j] = gsdata["SxSx"][(i,j)]
#     SySy[i,j] = gsdata["SySy"][(i,j)]
#     SzSz[i,j] = gsdata["SzSz"][(i,j)]
# end
FSxSx,FSySy,FSzSz = map(x -> FT2(getCorrMat(Latt,gsdata[(x,)],1/4;selected_point = 1:size(Latt)),Latt,lstk),[("Sx","Sx"),("Sy","Sy"),("Sz","Sz")])
FSS = FSxSx .+ FSySy .+ FSzSz
# FSS = FSS / maximum(FSS)
lsFFTS[i,j] = sum(FSS)
# Sm = maximum(FSxSx .+ FSySy .+ FSzSz)
# Smd = maximum(vcat(FSxSx,FSySy,FSzSz))*1.
# lsSSFp[i] = FSS[1]
end

figsize = (width = 300,height = 300)

fig = Figure()
axt = Axis(fig[1,1];figsize...,
xlabel = L"h",
ylabel = L"-\Gamma",
title = "size = $(size(Latt)), D = $(D), K = $(K), Γ = $(Γ)\n θ = $(θ/pi)π, ϕ = $(ϕ/pi)π",
# xlabel = L"k_x\ /\ \left(2\pi\sqrt{3}/3\right)",ylabel = L"k_y\ /\ 2\pi",
# xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),
# title = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle"
)


hmt = heatmap!(axt,lsHf,-lsΓ,lsFFTS;colormap = :Reds,colorrange = (3,5))
# boundary!(axt,FBZpoint;color = :white,linewidth = 1.5,linestyle = :dash)
# boundary!(axt,MFBZpoint;linewidth = 3,breathing=1.015)

Colorbar(fig[1,2],hmt;label = L"\tilde{S}(\Gamma)")


resize_to_layout!(fig)
display(fig)

save("$(figurename)/structure factor_h_G_point_Gamma_$(L)_$(D)_$(θ/pi)_$(ϕ/pi).png",fig)
save("$(figurename)/structure factor_h_G_point_Gamma_$(L)_$(D)_$(θ/pi)_$(ϕ/pi).pdf",fig)

# end

