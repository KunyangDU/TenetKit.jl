using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")
include("model.jl")

dataname = "NCTO/Kitaev/ED/data"


Lx = 2
Ly = 2

Latt = PCHoneyComb(Lx,Ly)

Sx = [0 1;1 0]/2
Sy = [0 -1im;1im 0]/2
Sz = [1 0;0 -1]/2

L = size(Latt)

xbonds,ybonds,zbonds = getxyzbonds(Latt;shift = [1/2,sqrt(3)/2], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])


K = 1.0

Kx = K
Ky = K 
Kz = K 

H = let H = zeros(2^L,2^L)

    for xb in xbonds
        H += addIntr2(xb,(Sx,Sx),L,Kx)
    end

    for yb in ybonds
        H += addIntr2(yb,(Sy,Sy),L,Ky)
    end

    for zb in zbonds
        H += addIntr2(zb,(Sz,Sz),L,Kz)
    end

    H
end


f = eigen(H)
Es = f.values
Vs = f.vectors

# H * 
mind = 114

# β = 100
# lsβ = easyinterp10((10. ^ (-0.2), 10 .^ (2.2)))
# lsβ = vcat(1.5 .^ (-15:1:-1),1:0.5:100)
# lsβ = vcat(2. .^ (-15:1:-1),1:100)
lsβ = vcat(1.5 .^ (-15:1:-1),1:0.5:100)
# lsβ = vcat(2. .^ (-15:1:-1),1:100)
lsβ = lsβ[2:end]*2
lsβ = lsβ[1:mind-1]
lsF = zeros(length(lsβ))
lsE = zeros(length(lsβ))
lsS = zeros(length(lsβ))
# lsC = zeros(length(lsβ) -1 )

for (i,β) in enumerate(lsβ)
Z = sum(@. exp( - β * Es))
lsF[i] = - log(Z) / β
lsE[i] = sum(@. exp( - β * Es) * Es) / Z
# lsC[i] = sum(@. exp( - β * Es) * Es) / Z
end

lsC = - lsβ[2:end] .^ 2 .* diff(lsE) ./ diff(lsβ)
lsS = (lsE - lsF) .* lsβ

figsize = (height = 150,width = 400)

fig = Figure()
# axF = Axis(fig[1,1])

axE = Axis(fig[1,1];figsize...,ylabel = L"E/N",
xscale = log10)
axS = Axis(fig[2,1];figsize...,ylabel = L"S/N",yticks = 0:0.2:1,
xscale = log10)
axC = Axis(fig[3,1];figsize...,ylabel = L"C/N",
xscale = log10)

lines!(axE, 1 ./ lsβ, lsE / size(Latt))
lines!(axS, 1 ./ lsβ, lsS / size(Latt) / log(2))
lines!(axC, 1 ./ lsβ[2:end], lsC / size(Latt))



for ax in [axC,axS,axE]
xlims!(ax,10. ^ (-2.2), 10 .^ (0.2))
end

ylims!(axS,0,1)

Label(fig[1,1][1, 1, Top()],"K = $(params1_Kitaev.K1), $(Ly)x$(Lx)x2 ZZ-HC-CY, D = $(D)\n$(paramsH)",
fontsize = 15,
font = :bold,
padding = (0, 0, 10, 0),
halign = :center
)

resize_to_layout!(fig)
display(fig)
eddata = Dict(
    "β" => lsβ,
    "E" => lsE,
    "F" => lsF,
    "S" => lsS,
    "C" => lsC
)

@save "$(dataname)/eddata_diff_$(Lx)x$(Ly)_K=$(K).jld2" eddata
