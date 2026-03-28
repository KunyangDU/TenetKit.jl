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

# for Hf in 1.04
Hf = 1.04

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)
Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
params = merge(params1_Kitaev,params_H)
params_Kitaev = merge(params1_Kitaev,params_H)
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
gsdata = gsdata["obs"]

Sxc,Syc,Szc = map(x -> [gsdata[((x,),)][((i,),)] for i in 1:size(Latt)], ["Sx","Sy","Sz"])
proj = collect(eachcol(hcat([1,-1,0]/sqrt(2),[1,1,-2]/sqrt(6),[1,1,1]/sqrt(3))'))
proj = collect(eachcol(hcat([1,-1,0]/sqrt(2),[1,1,-2]/sqrt(6),[1,1,1]/sqrt(3))))

Sx,Sy,Sz = map(x -> x[1]*Sz + x[2]*Sy + x[3]*Sz, proj)



figsize = (width = Lx*sqrt(3)*50,height = Ly*60)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)),
title = "$(2Lx)x$(2Ly) ZZ-HC-CY, zy, D=$(D)")
plotLatt!(ax,Latt,[0,1];site = false,sitelabel = false)
intensity = 100
# intensity = 0.45 / maximum(sum(sqrt.(Sy.^2 + Sx.^2))/length(Sy))
# colors = get(colorschemes[:bwr],Sy,(-Sm,Sm))
for i in 1:size(Latt)
    arrowc!(ax,coordinate(Latt,i)...,intensity * Sx[i],intensity * Sy[i];linewidth = 1.5)
end


resize_to_layout!(fig)
display(fig)

# Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr,label = L"\langle S_y\rangle")

Sx .^2 + Sy .^2 + Sz .^2

