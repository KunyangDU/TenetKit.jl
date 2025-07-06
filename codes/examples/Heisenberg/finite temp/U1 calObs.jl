using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/U1"

D = 2^7
Lx = 14
Ly = 1
params = (Jz = 1,Jxy = 0.5,h=0)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
lsβ2 = 2 * lsβ[2:end]
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

Obs = Observable()
LocalSpace = U₁Spin

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
end

for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
end

H = U1Hamiltonian(Latt; params...)

Es = lsE
Fs = lsF
E2s = zeros(length(lsβ2))
for (iβ,β) in enumerate(lsβ2)
    @show iβ/length(lsβ2)
    ρ = lsρ[iβ]
    ρH,_ = mul!(deepcopy(ρ),ρ,H;trunc = truncdim(D))
    E2s[iβ] = tr(ρH,ρH')
    calObs!(Obs,ρ;destroy = false)
    obs[iβ] = Obs.values
end

data = Dict(
    "E" => Es,
    "F" => Fs,
    "obs" => obs,
    "E2" => E2s
)

@save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
