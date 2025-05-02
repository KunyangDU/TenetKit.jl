
using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

totalname = "examples/ShastrySuther/data"
tailname = "SU2"

Lx = 4
Ly = 4
Latt = YCSS(Lx,Ly)
@save "$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], size(Latt)-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end

D = 2 ^ 9
lsλ = vcat(0.61:0.01:0.64,0.71:0.01:0.74,0.76:0.01:0.79)
for λ in lsλ
    params = (J1 = (λ == 0 ? 1e-8 : λ), J2 = 1)
    H = SU2Hamiltonian(Latt;params...)
    lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12), N = 6)
    showQuantSweep(lsEg ./ size(Latt))
    @save "$(totalname)/lsE_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" lsEg
    @save "$(totalname)/ψ_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" ψ
end




