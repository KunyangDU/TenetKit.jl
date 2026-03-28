using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/NCTO/Kitaev/data"
figurename = "NCTO/Kitaev/figures"
tailname = ""

D = 512
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params1_Kitaev = (J1 = 0.0, K1 = -1.0, Γ1 = 0.0, Γ1′ = 0.0)
params23 = (J2xy = 0.0, J3xy = 0.0, J3z = 0.0)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

mind = 65

# lsβ = vcat(1.5 .^ (-15:1:-1),1:0.5:100)
lsβ = vcat(2. .^ (-15:1:-1),1:100)
lsβ2 = lsβ[2:end]*2
lsβ2eff = lsβ2[1:mind-1]

# inputHxy = 0.04
lsHxy = vcat(0.0,0.04,0.12:0.04:0.4)

Stotal = zeros(length(lsHxy),mind-1)

for (i,Hxy) in enumerate(lsHxy)
Hx = 0.0
Hy = 0.0
Hz = Hxy
paramsH = (Hx = Hx, Hy = Hy, Hz = Hz)
params = merge(params1,params23,paramsH)
params_Kitaev = merge(params1_Kitaev,params23,paramsH)

# @load "$(dataname)/lsβ_$(Lx)x$(Ly).jld2" lsβ


lsE = zeros(mind-1)
lsF = zeros(mind-1)
for i in 2:mind
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_$(i).jld2" data
    lsE[i-1] = data["E"]
    lsF[i-1] = data["F"]
end

# lsC = - lsβ2eff[2:end] .^ 2 .* diff(lsE) ./ diff(lsβ2eff)
lsS = (lsE - lsF) .* lsβ2eff

Stotal[i,:] = lsS
end
Stotal /= log(2) * size(Latt)

figsize = (height = 300,width = 300)

fig = Figure()
# axF = Axis(fig[1,1])

axS = Axis(fig[1,1];figsize...,
ylabel = L"T",
xlabel = L"B_c",
# yticks = 0:0.2:1,
yscale = log10)

# heatmap!(axS,lsHxy, 1 ./ lsβ2eff, Stotal)
contourf!(axS,lsHxy, 1 ./ lsβ2eff, Stotal ,levels = 30)
ylims!(axS, 10. ^ (-2.), 10 .^ (0.))

Colorbar(fig[1,2], colorrange = (0,1),
label = L"S_m/N/\ln{2}")

Label(fig[1,1][1, 1, Top()],"K = $(params1_Kitaev.K1), $(Ly)x$(Lx)x2 ZZ-HC-CY, D = $(D)\n",
fontsize = 15,
font = :bold,
padding = (0, 0, 10, 0),
halign = :center
)

resize_to_layout!(fig)
display(fig)


save("$(figurename)/S_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).png",fig)
save("$(figurename)/S_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).pdf",fig)
