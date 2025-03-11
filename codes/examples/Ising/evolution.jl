
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")


Lx = 11
Ly = 1
D = 30

Latt = YCSqua(Lx,Ly)

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    PhySpace = TrivialSpinOneHalf.PhySpace 
    randMPS(PhySpace,AuxSpace)
end

params = (J=0,h=0,hz=1)

H = Hamiltonian(Latt;params...)
lsE = DMRG2!(ψ,H,truncdim(D) & truncbelow(1e-6);Nsweep=3)

params = (J=1,h=1,hz=0)

H = Hamiltonian(Latt;params...)
T = 6/params.J
Nt = 20
lst = range(0,T,Nt)

# lsψ, lst = TDVP2!(deepcopy(ψ), H, T, Nt, truncdim(D) & truncbelow(1e-6))
lst, lsψ, lsinfo = TDVP2!(ψ,H,T,Nt,D)
Szm = zeros(length(lst),size(Latt))
for ind in eachindex(lsψ)
    begin
        Obs = MPSObservable()
        LocalSpace = TrivialSpinOneHalf
        for i in 1:size(Latt)
            addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
        end
        calObs!(Obs, lsψ[ind])
    end   
    Szs = [Obs.values["Sz"][(i,)] for i in 1:size(Latt)]
    Szm[ind,:] = Szs
end
data = Dict(
    "Szm" => Szm,
    "lst" => lst,
)
@save "examples/Ising/data/data_evolve_D=$(D)_$(Lx)x$(Ly).jld2" data

