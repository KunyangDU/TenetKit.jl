using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/trivial"

D = 2^7
Lx = 10
Ly = 1
params = (J=1,)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
lsβ2 = 2 * lsβ[2:end]
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

Obs = Observable()
LocalSpace = TrivialSpinOneHalf

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    addObs!(Obs,LocalSpace.Sy,i,"Sy",nothing)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    addObs!(Obs,LocalSpace.Sz2,i,"Sz2",nothing)
end

for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),nothing)
    addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),nothing)
    addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
end 


H = TrivialHamiltonian(Latt; params...)

Es = lsE
Fs = lsF

E2s = zeros(length(lsβ2))
obs = repeat([Dict(),],length(lsβ2))

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
    "E2" => E2s,
    "obs" => obs
)

@save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
