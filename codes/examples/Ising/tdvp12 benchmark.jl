
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 11
Ly = 1
D = 50

Latt = YCSqua(Lx,Ly)

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    PhySpace = TrivialSpinOneHalf.PhySpace 
    randMPS(PhySpace,AuxSpace)
end

params = (J=0,h=0,hz=1)

H,r = Hamiltonian(Latt;params...)
lsE = DMRG1!(ψ,H,D,1e-6;cbe=true,Nsweep=3)

params = (J=1,h=1,hz=0)

H,r = Hamiltonian(Latt;params...)
T = 6/params.J
Nt = 20

lsψ, lst = TDVP1!(deepcopy(ψ), H, T, Nt, D)
Szm = zeros(length(lst),size(Latt),2)
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
    Szm[ind,:,1] = Szs
end

lsψ, lst = TDVP2!(deepcopy(ψ), H, T, Nt, D)
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
    Szm[ind,:,2] = Szs
end
cind = div(size(Latt),2) + 1
error12 = [sum(abs.(Szm[i,:,1] - Szm[i,:,2])) for i in eachindex(lst)] / size(Latt)
errorlr1 = [sum(abs.(Szm[i,1:cind-1,1] - Szm[i,end:-1:cind+1,1])) for i in eachindex(lst)]
errorlr2 = [sum(abs.(Szm[i,1:cind-1,2] - Szm[i,end:-1:cind+1,2])) for i in eachindex(lst)]
data = Dict(
    "lst" => lst,
    "error12" => error12,
    "errorlr1" => errorlr1,
    "errorlr2" => errorlr2,
    "Szm" => Szm,
    "lsψ" => lsψ
)
@save "examples/Ising/data/data_D=$(D)_$(Lx)x$(Ly).jld2" data

