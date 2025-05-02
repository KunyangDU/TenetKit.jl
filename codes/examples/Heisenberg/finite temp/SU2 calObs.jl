using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/SU2"

D = 2^9
Lx = 4
Ly = 4
params = (J=1,)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
lsβ2 = 2 * lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2




# H = SU2Hamiltonian(Latt; params...)
# M2 = SU2M2(Latt)


# Es = lsE
# Fs = lsF
# M2s = zeros(length(lsβ))
# E2s = zeros(length(lsβ))
# for (iβ,β) in enumerate(lsβ2)
#     @show iβ/length(lsβ)
#     ρ = lsρ[iβ]
#     # Z = exp(lsβ2[iβ] * lsF[iβ])
#     Z = iβ == 1 ? tr(ρ) : 1
#     # @show Z,exp(-lsβ2[iβ] * lsF[iβ])
#     ρH,_ = mul!(deepcopy(ρ),ρ,H)
#     ρM2,_ = mul!(deepcopy(ρ),ρ,M2)

#     E2s[iβ] = tr(ρH,ρH') / Z
#     M2s[iβ] = tr(ρ,ρM2') / Z
# end

# data = Dict(
#     "E" => Es,
#     "F" => Fs,
#     "M2" => M2s,
#     "E2" => E2s
# )

# @save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
