using TensorKit
include("../../src/iMPS.jl")
include("model.jl")



Lx = 4
Ly = 4

Latt = YCSqua(Lx,Ly)
Ndop = 0

ψ = let
    AuxSpace = vcat(Rep[U₁](Ndop // 2 => 1), repeat([Rep[U₁](i => 1 for i in -(abs(Ndop) + 1):1//2:(abs(Ndop)+1)),], size(Latt) - 1))
    randMPS(U₁Fermion.PhySpace, AuxSpace)
end

μ = 0
H = Hamiltonian(Latt;μ=μ)

D = 60
lsE = DMRG1!(ψ,H,truncdim(D)&truncbelow(1e-6);Nsweep=5)
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
Eg = lsE[end]

A = ψ.ts[div(size(Latt),2)].A
let 
    D = 0
    DD = 0
    for (c,b) in blocks(A)
        λ = diag(b)
        D += length(λ)
        DD += length(λ)*dim(c)
    end
    D,DD
end



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


