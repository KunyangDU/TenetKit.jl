
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 6
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

lsE = DMRG1!(ψ, H, truncdim(D) & truncbelow(1e-6);Nsweep=5)
showQuantSweep(lsE ./ N .- 1/4)
# @time "calculate observables" begin
#     Obs = MPSObservable()
#     LocalSpace = SU₂Spin

#     for pair in neighbor(Latt)
#         addObs!(Obs,LocalSpace.SS,pair,("S","S"),nothing)
#     end

#     calObs!(Obs, ψ)
# end

# gsObs = Obs.values
# gsψ = ψ

# @save "examples/SU2Spin/data/gsψ_$(Lx)x$(Ly)_$(D).jld2" gsψ
# @save "examples/SU2Spin/data/gsObs_$(Lx)x$(Ly)_$(D).jld2" gsObs

A = ψ.ts[div(size(Latt),2)].A
let 
    D = 0
    DD = 0
    for (c,b) in blocks(A)
        @show c,dim(c)
        λ = diag(b)
        D += length(λ)
        DD += length(λ)*dim(c)
    end
    D,DD
end

