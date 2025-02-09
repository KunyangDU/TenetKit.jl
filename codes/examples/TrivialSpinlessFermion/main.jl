using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
D = 2^3
μ = 0
H = Hamiltonian(Latt;μ=μ)

ψ, lsE = DMRG2!(ψ,H,D;LanczosLevel = 30,Nsweep = 1)
#showQuantSweep(lsE .- sum(@. -2cos(pi*(1:div(Lx,2))/(Lx+1))))
#@time "calculate observables" 
begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinlessFermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

#=     for k in -π:π/4:π
        addObs!(Obs.forest, (LocalSpace.F⁺F,LocalSpace.FF⁺,LocalSpace.n), Latt, [k,0], (("Fₖ⁺","Fₖ"),("Fₖ","Fₖ⁺"),"n"),LocalSpace.Z)
    end =#
    calObs!(Obs,ψ)
end
ntotal =  sum([Obs.values["n"][(i,)] for i in 1:size(Latt)])
Obs.values


