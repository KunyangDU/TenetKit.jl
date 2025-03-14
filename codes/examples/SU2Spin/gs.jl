
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

J = 1
D = 2^10
lsLx = [10,12]
Ly = 1
@save "examples/SU2Spin/data/lsLx_$(Ly).jld2" lsLx

for (i,Lx) in enumerate(lsLx)
    N = Lx*Ly

    ψ = let 
        AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], Lx*Ly-1))
        randMPS(SU₂Spin.PhySpace ,AuxSpace)
    end

    Latt = YCSqua(Lx,Ly)
    @save "examples/SU2Spin/data/Latt_$(Lx)x$(Ly).jld2" Latt

    H = Hamiltonian(Latt,J)

    lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-6),N = 3)
    showQuantSweep(lsEg ./ N .- 1/4)
    @save "examples/SU2Spin/data/lsE_$(Lx)x$(Ly).jld2" lsEg
end

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



