using TensorKit,CairoMakie,JLD2
include("../../src/iMPS.jl")
include("model.jl")

Ly = 1
Lx = 20
lsD = collect(600:200:800)

lsEerr = []
lsbtrial = []
bdata = Dict()
@save "examples/TrivialSpinlessFermion/data/bdata_lsD=$(lsD)_Lx=$(Lx)_Ly=$(Ly).jld" bdata

for (iD,D) in enumerate(lsD)
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
    "D" => lsD,
    "Lx" => Lx,
    "Ly" => Ly,
    "energyerror" => lsEerr,
    "trial" => lsbtrial
)

@save "examples/TrivialSpinlessFermion/data/bdata_lsD=$(lsD)_Lx=$(Lx)_Ly=$(Ly).jld" bdata









