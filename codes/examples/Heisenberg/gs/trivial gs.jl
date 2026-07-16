using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"
axpy!(::Number, ::Nothing, ::Nothing) = nothing
D = 256
Lx = 6
Ly = 6
params = (J = 1.0 ,J′ = 0.0,Hy = 0.0)

Latt = YCRect(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = TrivialHamiltonian(Latt;params...)

lsEg,lsinfo = DMRG2!(ψ, H; trunc = truncdim(D), N = 10)
# @save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

