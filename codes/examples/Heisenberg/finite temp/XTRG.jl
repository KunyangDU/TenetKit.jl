using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/XTRG"

tailname = "_1"

D = 64
Lx = 8
Ly = 1
Ds = 32
τ = 1.0

Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params = (J = 1.0, Δ = 1.0, Hz = 0.0)

H = TrivialHamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

N = 25

β0 = 2. ^ (-20)
lsβ = [β0 * 2^i for i in 1:N] * 2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ

SETTN1!(β0, H, ρ;trunc = truncdim(Ds))
ρ,lsinfo = XTRG1!(ρ,H,N;trunc = truncdim(D) & truncbelow(1e-16))
lsF = - map(x -> x.lnZ,lsinfo) ./ lsβ
lsE = map(x -> x.E,lsinfo)
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsE

