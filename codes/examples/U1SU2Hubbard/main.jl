using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
foldername = "examples/U1SU2Hubbard/data"

Lx = 4
Ly = 4

Latt = YCSqua(Lx,Ly)
Ndop = 0
@save "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let
    AuxSpace = vcat(Rep[U₁×SU₂]((Ndop, 0) => 1), repeat([Rep[U₁×SU₂]((i, j) => 1 for i in -(abs(Ndop) + 1):(abs(Ndop)+1) for j in 0:1//2:1),], size(Latt) - 1))
    randMPS(U₁SU₂Fermion.PhySpace, AuxSpace)
end

#= for U in[0,8]
    params = (U = U,)

    H = Hamiltonian(Latt;params...)

    D = 2^10

    lsE = DMRG2!(ψ,H,D)
    showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
    @save "$(foldername)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
end
 =#
#= @time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = U₁SU₂Fermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

    calObs!(Obs,ψ)
end

@show sum([Obs.values["n"][(i,)] for i in 1:size(Latt)])
Obs.values =#

params = (U = 0,)

H = Hamiltonian(Latt;params...)

D = 800

lsE,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-6))
@show lsE
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))


