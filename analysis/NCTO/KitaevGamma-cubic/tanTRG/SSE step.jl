using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC"
tailname = ""

D = 100
DS = 2^4

Lx = 3
Ly = 3

selected_point = 2Ly+1:size(Latt)-2Ly
selected_point = 1:size(Latt)

Nmax = 45

θ = 0.0 * pi
ϕ = 0.0 * pi

K = 1.0
J = 0.0
Γ = 0.0
Γ′ = 0.0
# for Γ′ in 0.01:0.01:0.09
params1_Kitaev = (J = J, K = K, Γ = Γ, Γ′ = Γ′)

Hf = 0.01
# for Hf in 0.02:0.04:0.8
Hx = Hf*sin(θ)*cos(ϕ)
Hy = Hf*sin(θ)*sin(ϕ)
Hz = Hf*cos(θ)

Hcx,Hcy,Hcz = Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
params_H_name = (Hx = round.(Hcx;digits = 3), Hy = round.(Hcy;digits = 3), Hz = round.(Hcz;digits = 3))

params_Kitaev = merge(params1_Kitaev,params_H)
params_Kitaev_name = merge(params1_Kitaev,params_H_name)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" lsβ2

Is = zeros(Nmax-1)

for i in 2:Nmax
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name)_$(i).jld2" data
    for dic in values(data["I"])
        effkeys = filter(x -> x[1][1] in selected_point, collect(keys(dic)))
        Is[i-1] += sum(map(x -> dic[x], effkeys))
    end
    # Is[i-1] = sum(map(x -> sum(values(x)),values(data["I"])))
end

figsize = (width = 400, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
figsize...,
xscale = log10,
yscale = log10
)

scatterlines!(ax,1 ./ lsβ2[1:Nmax-1],abs.(Is))
xlims!(ax,10.0 ^ (-2), 10.0 ^ (0))
# xlims!(ax,1e-8,0.5)

resize_to_layout!(fig)
display(fig)

# Is
lsβ2[Nmax-1]