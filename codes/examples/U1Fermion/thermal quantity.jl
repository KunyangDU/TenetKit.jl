using TensorKit,JLD2
include("../../src/iMPS.jl")
include("model.jl")

Lx = 4
Ly = 4
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 100
params = (μ=0,)

H= Hamiltonian(Latt;params...)

@load "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
f = zeros(length(lsβ))
u = zeros(length(lsβ))
u2 = zeros(length(lsβ))
lsβ *= 2

for (i,ρ) in enumerate(lsρ)
    @show i/length(lsρ)
    Z = tr(ρ)
    f[i] = -log(Z) / lsβ[i]
    u[i] = tr(ρ, H) / Z
    u2[i] = tr(mul!(deepcopy(ρ),ρ,H,1,0)) / Z
end
Ce = @. (u2 - u^2) * lsβ ^ 2
cβ = easyinterp10(lsβ)

f / L .- fe.(lsβ,Lx,Ly)
