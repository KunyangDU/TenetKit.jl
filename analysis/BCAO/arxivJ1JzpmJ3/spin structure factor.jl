using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/BCAO/arxivJ1JzpmJ3/data"
figurename = "BCAO/arxivJ1JzpmJ3/figures"
D = 2^6
Lx = 4
Ly = 4
params = (J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
# params = (hy = 1.0, J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg


lskx = 4pi/sqrt(3)*range(-1,1,51)
lsky = 2*0.999*pi*range(-1,1,51)
lsk = filter(x -> isinside(x,MFBZpoint;isboundary = true),[[kx,ky] for kx in lskx,ky in lsky][:])
lstk = map(x -> Tuple(x),lsk)

function getCorrMat(Latt::AbstractLattice,data::Dict,dataonsite = nothing;selected_point = 1:size(Latt))
    L = length(selected_point)
    A = zeros(L,L)
    for i in 1:L,j in i+1:L
        A[i,j] = data[(selected_point[i],selected_point[j])]
    end
    A = A + A'
    if !isnothing(dataonsite)
        if typeof(dataonsite) <: Dict
            for i in 1:L 
                A[i,i] = dataonsite[(i,)]
            end
        elseif typeof(dataonsite) <: Vector
            A .+= diagm(dataonsite)
        elseif typeof(dataonsite) <: Number 
            A .+= diagm(dataonsite*ones(L))
        end
    end
    return A
end

# SxSx1 = getCorrMat(Latt,gsdata["SxSx"],1/4)

# SxSx = zeros(size(Latt),size(Latt))
# SySy = zeros(size(Latt),size(Latt))
# SzSz = zeros(size(Latt),size(Latt))
# for i in 1:size(Latt),j in i+1:size(Latt)
#     SxSx[i,j] = gsdata["SxSx"][(i,j)]
#     SySy[i,j] = gsdata["SySy"][(i,j)]
#     SzSz[i,j] = gsdata["SzSz"][(i,j)]
# end

FSxSx,FSySy,FSzSz = map(x -> FT2(getCorrMat(Latt,gsdata[x],1/4),Latt,lstk),["SxSx","SySy","SzSz"])

x = map(lsk) do k
    k[1]
end
y = map(lsk) do k
    k[2]
end

Sm = maximum(FSxSx .+ FSySy .+ FSzSz)
Smd = maximum(vcat(FSxSx,FSySy,FSzSz))
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
hmt = heatmap!(axt,x,y,FSxSx .+ FSySy .+ FSzSz,colorrange = (0,Sm))
hmx = heatmap!(axx,x,y,FSxSx,colorrange = (0,Smd))
hmy = heatmap!(axy,x,y,FSySy,colorrange = (0,Smd))
hmz = heatmap!(axz,x,y,FSzSz,colorrange = (0,Smd))

# L"-\frac{2\pi}{3}",L"0",L"\frac{2\pi}{3}",L"\frac{4\pi}{3}"

for ax in [axt,axx,axy,axz]
    boundary!(ax,FBZpoint;color = :white,linewidth = 1.5,linestyle = :dash)
    boundary!(ax,dZZFBZpoint;color = :red,linewidth = 1.5)
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

Label(fig[1,1:2][1, 1, Top()], "Structure factor, $(Ly)x$(Lx)x2 ZZ-HC-CY, D = $(D)",
fontsize = 15,
font = :bold,
padding = (0, 0, 25, 0),
halign = :center)

resize_to_layout!(fig)
display(fig)

save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("$(figurename)/structure factor_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

