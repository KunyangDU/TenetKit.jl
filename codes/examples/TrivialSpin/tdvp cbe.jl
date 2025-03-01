
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 10
Ly = 1
Latt = YCSqua(Lx,Ly)
@save "examples/TrivialSpin/data/Latt_$(Lx)x$(Ly).jld2" Latt

J = 1
H = Hamiltonian(Latt;J=J)

D = 2^10
ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], Lx*Ly-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end
#@load "examples/SU2Spin/data/gsψ_$(Lx)x$(Ly)_$(D).jld2" gsψ

T = 2/J
Nt = 20

lsψ, lst = TDVP2!(gsψ, H, T, Nt, D)


Obs = MPSObservable()
LocalSpace = SU₂Spin
for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SS,(i,j),("S","S"),nothing)
end

lsObs = Vector{Dict}(undef,length(lsψ))

for (ind,ψi) in enumerate(lsψ)
    @show ind
    @time "calculate observables" begin
        calObs!(Obs, ψi; destroy = false)
    end
    lsObs[ind] = Obs.values
end

@save "examples/SU2Spin/data/lst_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_FM.jld2" lst
@save "examples/SU2Spin/data/lsψ_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_FM.jld2" lsψ
@save "examples/SU2Spin/data/lsObs_$(Lx)x$(Ly)_$(D)_$(round(T;digits=3))_$(Nt)_FM.jld2" lsObs

