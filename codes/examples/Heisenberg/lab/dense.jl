using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/lab/data"

D = 100
Lx = 64
Ly = 1
params = (J = 1, Δ = 0, Hx = 0,Hz = 0, hx = 0.1)

Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = TrivialHamiltonian(Latt;params...)

ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end
trunc = (truncdim(10) & truncbelow(1e-8))
Alg = CBEalgo(dynamicSVD(1.2,2),NoStruc(),0,_getdim(trunc))
multol = 1e-12
algo = Algebraalgo(SingleSite(),Alg,trunc,3,multol)
Hd = mul!(ρ,ρ,H,1,algo)[1]

# lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 100)
# @save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
# @save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# end
# lsEg,lsinfo = DMRG2!(ψ, Hd;trunc = truncdim(D) & truncbelow(1e-12),N = 100)
1