using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

# λ = 0.5
totalname = "examples/J1J2chain/data/rescale"
Ly = 1
tailname = "SU2"
lsλ = vcat(0.2:0.2:2,2.5:0.5:5,6:10)
lsλ = vcat(-reverse(lsλ),0,lsλ)
lsλ = [-1.5,-0.5,0.5,1.5]
lsLx = [20,]
for Lx in lsLx
    @show Lx
    N = Lx*Ly
    D = 2 ^ 9

    ψ = let 
        AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], Lx*Ly-1))
        randMPS(SU₂Spin.PhySpace ,AuxSpace)
    end

    Latt = YCSqua(Lx,Ly)
    @save "$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

    for λ in lsλ
        @show λ
        params = (J1 = λ, J2 = 1)
        H = Hamiltonian(Latt;params...)
        lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 6)
        showQuantSweep(lsEg ./ N)
        @save "$(totalname)/lsE_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" lsEg
        @save "$(totalname)/ψ_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" ψ
    end
end
