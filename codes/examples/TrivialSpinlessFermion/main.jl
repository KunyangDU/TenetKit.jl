using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 4
Ly = 4
D = 100

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)

lsE = DMRG1!(ψ,H,D,1e-6;Nsweep = 5,return_error = false)
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
#@time "calculate observables" 
#= begin
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
Obs.values =#

#= ρ = let 
    AuxSpaces = repeat([ℂ^1,], Lx*Ly+1)
    #ρ = IdDenseMPO(TrivialSpinlessFermion.PhySpace, AuxSpaces)
    ρ = RandDenseMPO(Lx*Ly,TrivialSpinlessFermion.PhySpace)
    canonicalize!(ρ,1)
    ρ
end
env = Environment([ρ,H,ρ'])
initialize!(env) =#

