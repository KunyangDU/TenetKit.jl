using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/SU2/connect"
# for D in [2^7,2^8]
D = 2^7
Lx = 14
Ly = 1
L = size(Latt)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J=1,)

H = SU2Hamiltonian(Latt; params...)
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
lnZ = - 2 * lsβ[end] * lsF[end]
ρ = lsρ[end]
lsβ = vcat(lsβ[end],2:10)
lsρ,lsinfo,lsF,lsE = tanTRG1!(ρ, H, lsβ;lnZ = lnZ,trunc = truncdim(D) & truncbelow(1e-12))


@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params)_connect_$(extrema(lsβ)).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)_connect_$(extrema(lsβ)).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)_connect_$(extrema(lsβ)).jld2" lsE
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_connect_$(extrema(lsβ)).jld2" lsβ

# end

