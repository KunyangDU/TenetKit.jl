using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/NCTO/Kitaev/data/XCHC"
figurename = "NCTO/Kitaev/figures/XCHC"
tailname = ""

D = 100
Lx = 4
Ly = 4
θ = 0.5 * pi
lsHB  = [0.05]
# lsϕ = 2pi*(0:0.1:0.9)
ϕ₀ = 0.0 * pi
lsϕ = ϕ₀ |> x -> x .+ 2pi*[-0.001,0.0,0.001]
HB = lsHB[1]
ϕ = lsϕ[1]

Hx = round(HB*sin(θ)*cos(ϕ);digits = 8)
Hy = round(HB*sin(θ)*sin(ϕ);digits = 8)
Hz = round(HB*cos(θ);digits = 8)

params1_Kitaev = (J1 = 0.0, K1 = -1.0, Γ1 = 0.0, Γ1′ = 0.0)
params23 = (J2xy = 0.0, J3xy = 0.0, J3z = 0.0)
paramsH = (Hx = Hx, Hy = Hy, Hz = Hz)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsH)
params_Kitaev = merge(params1_Kitaev,params23,paramsH)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/lsβ_$(Lx)x$(Ly).jld2" lsβ
mind = 25


# lsβ = vcat(1.5 .^ (-15:1:-1),1:0.5:100)
lsβ = vcat(2. .^ (-15:1:-1),1:100)
lsβ2 = lsβ[2:end]*2
lsβ2eff = lsβ2[1:mind-1]

lsE = zeros(mind-1)
lsF = zeros(mind-1)
for i in 2:mind
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_$(i).jld2" data
    lsE[i-1] = data["E"]
    lsF[i-1] = data["F"]
end

lsC = - lsβ2eff[2:end] .^ 2 .* diff(lsE) ./ diff(lsβ2eff)
lsS = (lsE - lsF) .* lsβ2eff

figsize = (height = 150,width = 400)

fig = Figure()
# axF = Axis(fig[1,1])

axE = Axis(fig[1,1];figsize...,ylabel = L"E/N",
xscale = log10)
axS = Axis(fig[2,1];figsize...,ylabel = L"S/N",yticks = 0:0.2:1,
xscale = log10)
axC = Axis(fig[3,1];figsize...,ylabel = L"C/N",
xscale = log10)

scatterlines!(axE,1 ./ lsβ2eff,lsE / size(Latt))
# lines!(axE,[1e-2,1e1],-9.100820134588293*ones(2) / size(Latt),linestyle = :dash,color = :grey)
# lines!(axE,[1e-2,1e1],-3.0958442172915*ones(2) / size(Latt),linestyle = :dash,color = :grey)
# lines!(axE,[1e-2,1e1],-3.818714554933768*ones(2) / size(Latt),linestyle = :dash,color = :grey)
# lines!(axE,[1e-2,1e1],-3.904505856014419*ones(2) / size(Latt),linestyle = :dash,color = :grey)
# lines!(axE,[1e-2,1e1],-1.5*ones(2) / size(Latt),linestyle = :dash,color = :grey)
# lines!(axE,[1e-2,1e1],-6.863432625749769*ones(2) / size(Latt),linestyle = :dash,color = :grey)
# lines!(axE,[1e-2,1e1],-5.9400438765228945*ones(2) / size(Latt),linestyle = :dash,color = :grey)

scatterlines!(axC,1 ./ lsβ2eff[2:end],lsC / size(Latt))
scatterlines!(axS,1 ./ lsβ2eff,lsS / size(Latt) / log(2))

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


save("$(figurename)/thermal_quant_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).png",fig)
save("$(figurename)/thermal_quant_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).pdf",fig)

# lsβ[35]
lsF / size(Latt)
