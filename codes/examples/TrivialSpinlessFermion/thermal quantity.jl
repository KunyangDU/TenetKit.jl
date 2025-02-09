using TensorKit, CairoMakie, LaTeXStrings
include("../../src/iMPS.jl")
include("model.jl")

function easyinterp10(v,N=100)
    return 10. .^ (range(log10.(extrema(v))..., N))
end

function ue(β,L)
    lsk = @. pi * (1:L) / (L+1)
    lsum = @.  ϵ(lsk) / (1 + exp( β * ϵ(lsk)))
    return sum(lsum) / L
end

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
#@load "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
D = 2^8

H = Hamiltonian(Latt;params...)

@load "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
f = zeros(length(lsβ))
u = zeros(length(lsβ))
lsβ *= 2

for (i,ρ) in enumerate(lsρ)
    Z = tr(ρ)
    f[i] = -log(Z) / lsβ[i]
    u[i] = tr(ρ, H) / Z
end

lsT1 = centralize(1 ./ lsβ)
lsβ1 = centralize(lsβ)
Ce = - centralize(lsβ) .* diff(u) ./ diff(log.(lsβ))

cβ = easyinterp10(lsβ)

figsize = (height=150,width=300)
fig = Figure()
axf = Axis(fig[1,1];xscale=log10,figsize...,
title = "Spinless free fermion",
ylabel = L"F\ /\ N" )
scatter!(axf, 1 ./ lsβ, f / L)
lines!(axf, 1 ./ easyinterp10(lsβ), fe.(easyinterp10(lsβ),L);color = :red)

axu = Axis(fig[2,1];xscale=log10,figsize...,
ylabel = L"U\ /\ N")
scatter!(axu, 1 ./ lsβ, u / L)
lines!(axu, 1 ./ cβ, ue.(cβ,L);color = :red)

axce = Axis(fig[3,1];xscale=log10,figsize...,
xlabel = L"T",ylabel =L"C_e\ /\ N")
scatter!(axce, lsT1, Ce / L)
lines!(axce, 1 ./ cβ, ce.(cβ,L);color = :red)

hidexdecorations!(axf;ticks = false,grid = false)
hidexdecorations!(axu;ticks = false,grid = false)

resize_to_layout!(fig)
display(fig)



save("examples/TrivialSpinlessFermion/figures/thermal quantity.png",fig)

f .- fe.(lsβ,L) * L
(Ce .- ce.(lsβ1,L) * L) ./ (ce.(lsβ1,L) * L)

