using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/Hubbard/data"


Lx = 4
Ly = 4
Ndop = 0
params = (U = 0,μ = 0)
D = 2^7

Latt = YCSqua(Lx,Ly)
println("$(Lx)x$(Ly), D = $(D), params = $(params)")
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let
    AuxSpace = vcat(Rep[U₁×SU₂]((Ndop, 0) => 1), repeat([Rep[U₁×SU₂]((i, j) => 1 for i in -(abs(Ndop) + 1):(abs(Ndop)+1) for j in 0:1//2:1),], size(Latt) - 1))
    randMPS(U₁SU₂Fermion.PhySpace, AuxSpace)
end

H = Hamiltonian(Latt;params...)


lsEg,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-12))
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg



