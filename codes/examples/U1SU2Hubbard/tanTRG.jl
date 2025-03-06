using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#
foldername = "examples/U1SU2Hubbard/data"
Lx = 8
Ly = 4
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2^11
for U in [0,8]
    params = (U = U,)
    H = Hamiltonian(Latt; params...)
    ρ = let 
        AuxSpaces = repeat([Rep[U₁×SU₂]((0, 0) => 1),], size(Latt)+1)
        ρ = IdDenseMPO(U₁SU₂Fermion.PhySpace, AuxSpaces)
        canonicalize!(ρ,1)
        ρ
    end

    lsβ = vcat(2. .^ (-10:1:-1), 1:10)

    SETTN!(lsβ[1], H, ρ;D=D)
    lsρ = tanTRG2!(ρ, H, lsβ, D)

    @save "$(foldername)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
    @save "$(foldername)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
end


