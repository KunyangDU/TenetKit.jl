
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1


ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace     ,AuxSpace)
end 

Latt = YCSqua(Lx,Ly)

D = 2^8
params = (J=1,h=0.2,hz=0)

H = Hamiltonian(Latt;params...)

lsE = DMRG2!(ψ,H,D)
Eg = lsE[end]
@save "examples/Ising/data/ψ.jld2" ψ
@save "examples/Ising/data/Eg.jld2" Eg

#= showQuantSweep(lsE ./(J*Lx) .-0.25)
@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf
    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",nothing)
    end

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for pair in neighbor(Latt)
        addObs!(Obs,LocalSpace.SxSx,pair,("Sx","Sx"),nothing)
        addObs!(Obs,LocalSpace.SySy,pair,("Sy","Sy"),nothing)
        addObs!(Obs,LocalSpace.SzSz,pair,("Sz","Sz"),nothing)
    end

    calObs!(Obs, ψ)
end

#@show sum(map(y -> Obs.values[y][round(Int64,Lx/2) |> x -> (x,x+1)],["SxSx","SySy","SzSz"]))
Obs.values =#
