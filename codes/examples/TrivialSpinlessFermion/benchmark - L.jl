using TensorKit,CairoMakie,JLD2
include("../../src/iMPS.jl")
include("model.jl")

Ly = 1
D = 2^5
lsLx = 20:10:100

lsEerr = []
lsbtrial = []

for (iLx,Lx) in enumerate(lsLx)
    Latt = YCSqua(Lx,Ly)
    H = Hamiltonian(Latt)
    ψ = let 
        AuxSpace = repeat([ℂ^1,], Lx*Ly)
        randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
    end

    lsE,btrial = bDMRG2!(ψ,H,D,1e-8;Nsweep = 16)
    Eerr = norm((lsE .- sum(@. -2 * cos.(pi*(1:div(Lx*Ly,2))/(Lx*Ly+1)))) ./ lsE)
    @show Eerr
    push!(lsEerr,Eerr)
    push!(lsbtrial,btrial)

end

bdata = Dict(
    "D" => D,
    "Lx" => lsLx,
    "Ly" => Ly,
    "energyerror" => lsEerr,
    "trial" => lsbtrial
)

@save "examples/TrivialSpinlessFermion/data/bdata_D=$(D)_lsLx=$(lsLx)_Ly=$(Ly).jld" bdata









