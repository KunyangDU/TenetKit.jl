using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/U1"

D = 2^9
Lx = 4
Ly = 4
params = (Jz = 1,Jxy = 0.5,h=0)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
lsβ2 = 2 * lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2


ρ = lsρ[end]

Obs = MPSObservable()
LocalSpace = U₁Spin

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
end

for pair in neighbor(Latt)
    addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
end

calObs!(Obs, ρ)

gsdata = Obs.values

# H = U1Hamiltonian(Latt; params...)
# Mz = U1Mz(Latt)
# # Mz2 = U1Mz2(Latt)

# Es = lsE
# Fs = lsF
# Mz2s = zeros(length(lsβ))
# Mzs = zeros(length(lsβ))
# E2s = zeros(length(lsβ))
# for (iβ,β) in enumerate(lsβ2[1:end-4])
#     @show iβ/length(lsβ)
#     ρ = lsρ[iβ]
#     # Z = exp(lsβ2[iβ] * lsF[iβ])
#     Z = iβ == 1 ? tr(ρ) : 1
#     # @show Z,exp(-lsβ2[iβ] * lsF[iβ])
#     ρH,_ = mul!(deepcopy(ρ),ρ,H)
#     ρMz,_ = mul!(deepcopy(ρ),ρ,Mz)
#     # ρMz2,_ = mul!(deepcopy(ρ),ρ,Mz2)

#     E2s[iβ] = tr(ρH,ρH') / Z
#     Mz2s[iβ] = tr(ρMz,ρMz') / Z
#     Mzs[iβ] = tr(ρ,ρMz') / Z
#     GC.gc()
# end

# data = Dict(
#     "E" => Es,
#     "F" => Fs,
#     "Mz2" => Mz2s,
#     "Mz" => Mzs,
#     "E2" => E2s
# )

# @save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
