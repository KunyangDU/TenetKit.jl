using TensorKit,JLD2
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 128
params = (μ=0,)

H= Hamiltonian(Latt;params...)

@load "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "examples/U1Fermion/data/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@load "examples/U1Fermion/data/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
# @load "examples/U1Fermion/data/H2_$(Lx)x$(Ly)_$(D)_$(params).jld2" H2

u = zeros(length(lsβ))
u2 = zeros(length(lsβ))
lsβ2 = 2 * lsβ

for (i,ρ) in enumerate(lsρ)
    @show i/length(lsρ)
    Z = -lsF[i] * lsβ2[i]
    u2[i] = tr(mul!(deepcopy(ρ),ρ,H)[1])
end
Ce = @. (u2 - lsE^2) * lsβ2 ^ 2

data = Dict(
    "f" => lsF,
    "u" => lsE,
    "Ce" => Ce
)

@save "examples/U1Fermion/data/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
@save "examples/U1Fermion/data/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

(lsF/L  .- fe.(lsβ2[1:length(lsF)],Lx,Ly)) ./ fe.(lsβ2[1:length(lsF)],Lx,Ly)
(lsE/L  .- ue.(lsβ2[1:length(lsE)],Lx,Ly)) ./ ue.(lsβ2[1:length(lsE)],Lx,Ly)
(Ce/L .- ce.(lsβ2[1:length(lsE)],Lx,Ly)) ./ ce.(lsβ2[1:length(lsE)],Lx,Ly)
