using TensorKit
include("../../src/iMPS.jl")
include("model.jl")


Lx = 6
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
#@load "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
D = 30

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

data = Dict(
    "f" => f,
    "u" => u,
    "Ce" => Ce
)

@save "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ
@save "examples/TrivialSpinlessFermion/data/data_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" data

f .- fe.(lsβ,Lx,Ly) * L

