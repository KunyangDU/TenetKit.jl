using TensorKit,JLD2
include("../../src/iMPS.jl")
include("model.jl")
foldername = "examples/U1SU2Hubbard/data"

Lx = 6
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@load "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 30
params = (U=0,)

H= Hamiltonian(Latt;params...)

@load "$(foldername)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(foldername)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
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
