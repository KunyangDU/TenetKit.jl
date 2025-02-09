
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 32
Ly = 1
N = Lx*Ly

ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], Lx*Ly-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
@save "examples/SU2Spin/data/Latt_$(Lx)x$(Ly).jld2" Latt

J = 1
D = 2^6

H = Hamiltonian(Latt,J)

ψ, lsE = DMRG2!(ψ, H, D;Nsweep=5,LanczosLevel = 25)
showQuantSweep(lsE ./ N .- 1/4)
@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = SU₂Spin

    for pair in neighbor(Latt)
        addObs!(Obs,LocalSpace.SS,pair,("S","S"),nothing)
    end

    calObs!(Obs, ψ)
end

gsObs = Obs.values
gsψ = ψ

@save "examples/SU2Spin/data/gsψ_$(Lx)x$(Ly)_$(D).jld2" gsψ
@save "examples/SU2Spin/data/gsObs_$(Lx)x$(Ly)_$(D).jld2" gsObs

