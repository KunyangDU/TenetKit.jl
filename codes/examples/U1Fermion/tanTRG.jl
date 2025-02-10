using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{2}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[3,-1,1,5] * B.A[6,3] * C.A[4,5,6,-3] * EnvR.A[1,-2,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{2},Er::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,2] * Er.A[1,-2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(El::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,-3,1] * A.A[2,1,-4,-5] * B.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{2}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,-2,1] * A.A[2,1,-3,5] * B.A[4,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{3, 7}, A::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-2,1,-4,-5,-6,-7] * A.A[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, A::CompositeMPOTensor{2, 6}, B::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-4,1] * A.A[-3,2,1,-5,-6,-7] * B.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

Lx = 3
Ly = 3
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2^8
params = (μ=0,)

H = Hamiltonian(Latt;params...)
ρ = let 
    AuxSpaces = repeat([Rep[U₁](0 => 1),],size(Latt)+1)
    ρ = IdDenseMPO(U₁Fermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-5:2:-1), 1:10)

SETTN!(lsβ[1], H, ρ;D=D)
lsρ = tanTRG2!(ρ, H, lsβ, D;LanczosLevel = 15,TruncErr=1e-2)

@save "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ

