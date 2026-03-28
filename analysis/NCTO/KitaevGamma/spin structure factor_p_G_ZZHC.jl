using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

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

dataname = "../codes/examples/NCTO/KitaevGamma/data/ZZHC"
figurename = "NCTO/KitaevGamma/figures/ZZHC"
tailname = ""

D = 2^8
Lx = 4
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)


θ = pi / 2
ϕ = pi / 2



K = 1
Γ = -0.32

params1_Kitaev = (J1 = J, K1 = K, Γ1 = Γ, Γ1′ = Γ1′)
params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

lsHf = 0:0.04:0.8
lsk = [coordinate([0,0];basis = KBASIS2),]
lsSSFp = zeros(length(lsHf))
for (i,Hf) in enumerate(lsHf)
Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)
params_H = (Hx = Hx, Hy = Hy, Hz = Hz)

params = merge(params1,params23,params_H)
params_Kitaev = merge(params1_Kitaev,params23,params_H)
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
FSxSx,FSySy,FSzSz = map(x -> FT2(getCorrMat(Latt,gsdata[(x,)],1/4;selected_point = 1:size(Latt)),Latt,lstk),[("Sx","Sx"),("Sy","Sy"),("Sz","Sz")])
FSS = FSxSx .+ FSySy .+ FSzSz

# Sm = maximum(FSxSx .+ FSySy .+ FSzSz)
# Smd = maximum(vcat(FSxSx,FSySy,FSzSz))*1.
lsSSFp[i] = FSS[1]
end

figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...)

scatterlines!(ax,lsHf[3:end],[lsSSFp[i + 2] + lsSSFp[i] - 2lsSSFp[i + 1] for i in 1:length(lsHf)-2])

xlims!(ax,extrema(lsHf))

resize_to_layout!(fig)
display(fig)

# save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).png",fig)
# save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).pdf",fig)

# end

