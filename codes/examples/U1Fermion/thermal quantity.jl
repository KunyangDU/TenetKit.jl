using TensorKit,JLD2
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 32
params = (μ=0,)

H= Hamiltonian(Latt;params...)

@load "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "examples/U1Fermion/data/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@load "examples/U1Fermion/data/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
f = zeros(length(lsβ))
u = zeros(length(lsβ))
u2 = zeros(length(lsβ))
lsβ2 = 2 * lsβ

for (i,ρ) in enumerate(lsρ)
    @show i/length(lsρ)
    Z = -lsF[i] * lsβ2[i]
    # u[i] = tr(ρ, H) / Z
    # u2[i] = tr(mul!(deepcopy(ρ),ρ,H,1,0)[1]) / Z
end
# Ce = @. (u2 - u^2) * lsβ2 ^ 2

data = Dict(
    "f" => lsF,
    "u" => lsE,
    # "Ce" => Ce
)

@save "examples/U1Fermion/data/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
@save "examples/U1Fermion/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

(lsF  .- fe.(lsβ2[1:length(lsF)],Lx,Ly) * L) ./ lsF
(lsE  .- ue.(lsβ2[1:length(lsE)],Lx,Ly) * L) ./ lsE

# lsF .- f

