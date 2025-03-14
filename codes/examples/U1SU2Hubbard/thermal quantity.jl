using TensorKit,JLD2
include("../../src/iMPS.jl")
include("model.jl")
foldername = "examples/U1SU2Hubbard/data"

Lx = 2
Ly = 4
Latt = iYCSqua(Lx,Ly)
L = size(Latt)
@load "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 400
params = (U=1e-3,μ=0)

H= Hamiltonian(Latt;params...)

@load "$(foldername)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(foldername)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "$(foldername)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@load "$(foldername)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE

u = zeros(length(lsβ))
u2 = zeros(length(lsβ))
lsβ2 = 2 * lsβ

for (i,ρ) in enumerate(lsρ)
    @show i/length(lsρ)
    # Z = -lsF[i] * lsβ2[i]
    # u[i] = tr(ρ, H)
    # u2[i] = tr(mul!(deepcopy(ρ),ρ,H,1,0)) / Z
end
# Ce = @. (u2 - u^2) * lsβ ^ 2
# cβ = easyinterp10(lsβ)

# lsF = (lsF/L  .- fe.(lsβ2[eachindex(lsF)],Lx,Ly)) * 0.2 * L + fe.(lsβ2[eachindex(lsF)],Lx,Ly)* L 
# lsE = (lsE/L  .- ue.(lsβ2[eachindex(lsF)],Lx,Ly) ) * 0.2 * L + ue.(lsβ2[eachindex(lsF)],Lx,Ly)* L 

data = Dict(
    "lsβ" => lsβ2,
    "u" => lsE,
    "f" => lsF
)

@save "$(foldername)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data


# (u/L  .- ue.(lsβ2[eachindex(lsF)],Lx,Ly) ) ./ ue.(lsβ2[eachindex(lsF)],Lx,Ly)
(lsE/L  .- ue.(lsβ2[eachindex(lsF)],Lx,Ly)) ./ ue.(lsβ2[eachindex(lsF)],Lx,Ly)
