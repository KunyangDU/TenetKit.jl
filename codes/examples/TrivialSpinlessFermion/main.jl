using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

Lx = 16
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
D = 200
μ = 0
H = Hamiltonian(Latt;μ=μ)

ψ, lsE = DMRG2!(ψ,H,D;LanczosLevel = 15,Nsweep = 20,return_error = false)
showQuantSweep(lsE)
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

ψ.ts[1]
ρ = let 
    AuxSpaces = repeat([ℂ^1,], Lx*Ly+1)
    ρ = IdDenseMPO(TrivialSpinlessFermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end
ρ.ts[1]

