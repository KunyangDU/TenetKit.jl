using TensorKit,CairoMakie,JLD2
include("../../src/iMPS.jl")
include("model.jl")

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{3}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    return @tensor EnvL.A[3,1] * A.A[2,1,6,5] * B.A[4,7,2] * C.A[8,5,4,3] * EnvR.A[6,7,8]
end

Lx = 3
Ly = 3
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2^8
params = (μ=0,)

H= Hamiltonian(Latt;params...)

@load "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
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
Ce = @. (u2 - u^2) * lsβ ^ 2
cβ = easyinterp10(lsβ)

figsize = (height=150,width=300)
fig = Figure()
axf = Axis(fig[1,1];xscale=log10,figsize...,
title = "U1 fermion",
ylabel = L"F\ /\ N" )
scatter!(axf, 1 ./ lsβ, f / L)
lines!(axf, 1 ./ cβ, fe.(cβ,Lx,Ly);color = :red)

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

save("examples/U1Fermion/figures/thermal_quant_$(Lx)x$(Ly)_D=$(D).pdf",fig)
save("examples/U1Fermion/figures/thermal_quant_$(Lx)x$(Ly)_D=$(D).png",fig)

f / L .- fe.(lsβ,Lx,Ly)
