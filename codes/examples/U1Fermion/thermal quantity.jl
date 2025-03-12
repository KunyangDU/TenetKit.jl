using TensorKit,JLD2
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 120
params = (μ=0,)

H= Hamiltonian(Latt;params...)

@load "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
f = zeros(length(lsβ))
u = zeros(length(lsβ))
u2 = zeros(length(lsβ))
lsβ2 = 2 * lsβ

for (i,ρ) in enumerate(lsρ)
    @show i/length(lsρ)
    Z = tr(ρ)
    f[i] = -log(Z) / lsβ2[i]
    u[i] = tr(ρ, H) / Z
    u2[i] = tr(mul!(deepcopy(ρ),ρ,H,1,0)[1]) / Z
end
Ce = @. (u2 - u^2) * lsβ2 ^ 2

data = Dict(
    "f" => f,
    "u" => u,
    "Ce" => Ce
)


@save "examples/U1Fermion/data/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
@save "examples/U1Fermion/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

f / L .- fe.(lsβ2,Lx,Ly)

