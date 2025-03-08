using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

function contract(EnvL::LeftCompositeEnvironmentTensor{2,5}, A::DenseMPOTensor{4})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[1,2,-3,-4,-5] * A'.A[3,4,2,1] * A.A[-2,-1,3,4])
end

function contract(EnvR::RightCompositeEnvironmentTensor{2,5}, B::DenseMPOTensor{4})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ EnvR.A[-1,-2,2,1,-5] * B'.A[1,4,2,3] * B.A[-3,3,-4,4])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, Λ::DenseMPOTensor{2})
    return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[-1,-2,-3,1,-5]*Λ.A[1,-4])
end

function contract(EnvR::RightCompositeEnvironmentTensor{2, 5}, Λ::DenseMPOTensor{2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ Λ.A[-1,1]*EnvR.A[1,-2,-3,-4,-5])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, A::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3,3] * A.A[-1,3,2,1] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{2, 5}, A::AdjointMPOTensor{4})
    @tensor tmp[-1 -2;-3] ≔ EnvR.A[-1,-2,2,1,3] * A.A[1,3,2,-3] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{2, 5})
    @tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,2,1] * EnvR.A[1,2,-1,-3,-4]
    return DenseMPOTensor(tmp)
end


#= 
Fermion complexity
=#

Lx = 4
Ly = 4
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 64
params = (μ=0,)

H = Hamiltonian(Latt;params...)
ρ = let 
    AuxSpaces = repeat([Rep[U₁](0 => 1),],size(Latt)+1)
    ρ = IdDenseMPO(U₁Fermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-10:1:-1), 1:10)

SETTN!(lsβ[1], H, ρ;D=D)
lsρ = tanTRG1!(ρ, H, lsβ, D)

@save "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ

