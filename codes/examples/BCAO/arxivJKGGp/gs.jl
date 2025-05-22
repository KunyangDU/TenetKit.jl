using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/BCAO/arxivJKGGp/data/pin"

D = 2^6
Lx = 6
Ly = 4
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=0.0,J3xy=0.0,J3z=0.0)
# params = (J1=-0.1,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.3,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# for J1 in vcat(0.45:0.01:0.49,0.51:0.01:0.55)
#     @show J1

# params = (J1=-0.46,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.008)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.50035,K1=-0.85893,Γ1=0.4538,Γ1′=0.09311,J2= -0.0321,J3xy=0.26,J3z=0.0078)
# for J1 in -0.513:-0.002:-0.515
params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.093,J2=-0.032,J3xy=0.26,J3z=0.0078)

println("$(Lx)x$(Ly), D = $(D), params = $(params)")
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

pinh = map(x -> x*1,vcat(repeat([[0.,1.,0.],],2Ly),repeat([[0.,-1.,0.],],2Ly)))
H = TrivialHamiltonian(Latt;params...,pinh = pinh)
# H = TrivialHamiltonian(Latt;params...)
lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# end

