using TensorKit
include("../../src/iMPS.jl")
include("model.jl")



Lx = 8
Ly = 4

Latt = YCSqua(Lx,Ly)
Ndop = 0

ψ = let
    AuxSpace = vcat(Rep[U₁](Ndop // 2 => 1), repeat([Rep[U₁](i => 1 for i in -(abs(Ndop) + 1):1//2:(abs(Ndop)+1)),], size(Latt) - 1))
    randMPS(U₁Fermion.PhySpace, AuxSpace)
end

μ = 0
H = Hamiltonian(Latt;μ=μ)
D = 120
# env = Environment([ψ,H,ψ'])
# initialize!(env)
# Alg = DMRGalgo(SingleSite(),CBEalgo(randSVD(),1.2),D,1e-6,5,1e-4,DMRGDefaultLanczos)
#Alg = DMRGalgo(DoubleSite(),NoAlgorithm(),D,1e-6,5,1e-4,DMRGDefaultLanczos)

lsE,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-6))

showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))

#= 


lsE = DMRG1!(ψ,H,truncdim(D)&truncbelow(1e-6);Nsweep=5)
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
Eg = lsE[end]


Env = Environment([ψ,H,ψ'])
typeof(Env) =#




#= @time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = U₁Fermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

    calObs!(Obs,ψ)
end

density = [Obs.values["n"][(i,)] for i in 1:size(Latt)]
 =# 


