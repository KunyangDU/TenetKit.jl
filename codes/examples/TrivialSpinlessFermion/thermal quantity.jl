using TensorKit, CairoMakie, LaTeXStrings
include("../../src/iMPS.jl")
include("model.jl")


Lx = 6
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
#@load "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
D = 2^6

tailname = "_tanTRG"

H = Hamiltonian(Latt;params...)

@load "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ
@load "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsρ
f = zeros(length(lsβ))
u = zeros(length(lsβ))
u2 = zeros(length(lsβ))
lsβ *= 2

for (i,ρ) in enumerate(lsρ)
    Z = tr(ρ)
    f[i] = -log(Z) / lsβ[i]
    u[i] = tr(ρ, H) / Z
    u2[i] = tr(mul!(deepcopy(ρ),ρ,H,1,0)) / Z
end

Ce = @. (u2 - u^2) * lsβ^2

cβ = easyinterp10(lsβ)

figsize = (height=150,width=300)
fig = Figure()
axf = Axis(fig[1,1];xscale=log10,figsize...,
title = "Spinless free fermion",
ylabel = L"F\ /\ N" )
scatter!(axf, 1 ./ lsβ, f / L)
lines!(axf, 1 ./ easyinterp10(lsβ), fe.(easyinterp10(lsβ),Lx,Ly);color = :red)

axu = Axis(fig[2,1];xscale=log10,figsize...,
ylabel = L"U\ /\ N")
scatter!(axu, 1 ./ lsβ, u / L)
lines!(axu, 1 ./ cβ, ue.(cβ,Lx,Ly);color = :red)

axce = Axis(fig[3,1];xscale=log10,figsize...,
xlabel = L"T",ylabel =L"C_e\ /\ N")
scatter!(axce, 1 ./ lsβ, Ce / L)
lines!(axce, 1 ./ cβ, ce.(cβ,Lx,Ly);color = :red)

hidexdecorations!(axf;ticks = false,grid = false)
hidexdecorations!(axu;ticks = false,grid = false)

resize_to_layout!(fig)
display(fig)



save("examples/TrivialSpinlessFermion/figures/thermal quantity_SETTN.png",fig)

f .- fe.(lsβ,Lx,Ly) * L

