using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/BCAO/arxivJ1JzpmJ3/data/pin"

D = 2^6
Lx = 4
Ly = 4
params = (
    J1xy = -1.1735687936596617,
    J1z = -0.42248476571747806,
    Jpm = 0.026992082254172278,
    Jzpm = -0.6689342123860069,
    J2 = -0.03755420139710917,
    J3xy = 0.30512788635151195,
    J3z = 0.009153836590545358
)

params = (J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
println("$(Lx)x$(Ly), D = $(D), params = $(params)")
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end
# pinh = map(x -> x*1,repeat([[0.,1.,0.],],4Ly))
pinh = vcat(repeat([[0.,1.,0.],],2Ly),repeat([[0.,-1.,0.],],2Ly))
H = TrivialHamiltonian(Latt;params...,pinh = pinh)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ



