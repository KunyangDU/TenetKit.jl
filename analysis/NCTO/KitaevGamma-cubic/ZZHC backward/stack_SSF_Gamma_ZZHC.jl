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
Lx = 3
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)


θ = 0.0 * pi
ϕ = 0.5 * pi


lsHf = 0:0.04:0.8
lsk = [coordinate([0,0];basis = KBASIS2),]
# lsΓ′ = vcat(0:0.02:0.08,0.12:0.04:0.2)
lsΓ′ = 0:0.04:0.2
lsSSFp = zeros(length(lsΓ′),length(lsHf))

K = 1.0
for (iΓ′,Γ′) in enumerate(lsΓ′)
params1_Kitaev = (K = K, Γ′ = Γ′)



for (i,Hf) in enumerate(lsHf)
Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)
Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params = merge(params1_Kitaev,params_H)
params_Kitaev = merge(params1_Kitaev,params_H)
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
gsdata = gsdata["obs"]


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
FSS = FSxSx .+ FSySy .+ FSzSz
# Sm = maximum(FSxSx .+ FSySy .+ FSzSz)
# Smd = maximum(vcat(FSxSx,FSySy,FSzSz))*1.
lsSSFp[iΓ′,i] = FSS[1]
end
end

figsize = (width = 200,height = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "$(Ly)x$(Lx)x4, D = $(D), Γ",
ylabel = L"\tilde{S}(\mathbf{k})",
xlabel = L"h")

# scatterlines!(ax,lsHf,lsSSFp)
for (i,Γ′) in enumerate(lsΓ′)
scatterlines!(ax,lsHf,lsSSFp[i,:] .+ i*0.2,label = "Γ′=$(Γ′)")
# scatterlines!(ax,lsHf,lsM)
end

xlims!(ax,extrema(lsHf))

Legend(fig[1,2],ax)

resize_to_layout!(fig)
display(fig)

save("$(figurename)/stack_SSF_$(Lx)x$(Ly)_$(D)_$(θ/pi)_$(ϕ/pi)_Gamma.png",fig)
save("$(figurename)/stack_SSF_$(Lx)x$(Ly)_$(D)_$(θ/pi)_$(ϕ/pi)_Gamma.pdf",fig)
# end
