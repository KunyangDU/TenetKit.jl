using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/triangle/trivial"

D = 2^6
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


Obs = MPSObservable()
LocalSpace = TrivialSpinOneHalf

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    addObs!(Obs,LocalSpace.Sy,i,"Sy",nothing)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
end

for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),nothing)
    addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),nothing)
    addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
end 

H = TrivialHamiltonian(Latt; params...)

E2s = zeros(length(lsβ))
obs = repeat([Dict(),],length(lsβ))

Es = lsE
Fs = lsF

for (iβ,β) in enumerate(lsβ2[1:28])
    @show iβ
    ρ = lsρ[iβ]
    Z = iβ == 1 ? tr(ρ) : 1
    E2s[iβ] = tr(ρH,ρH') / Z
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
