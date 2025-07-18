using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Kitaev_XCHC/data"
D = 2^9
D1 = D
D2 = D
d = 2

C = composite(DenseMPOTensor(randn,ℂ^d ⊗ ℂ^D1, ℂ^D2 ⊗ ℂ^d),DenseMPOTensor(randn,ℂ^d ⊗ ℂ^D1, ℂ^D2 ⊗ ℂ^d))
# C = composite(MPSTensor(randn,ℂ^D1⊗ℂ^d, ℂ^D2),MPSTensor(randn,ℂ^D1⊗ℂ^d, ℂ^D2))

LocalSpace = TrivialSpinOneHalf

op = LocalOperator(LocalSpace.Sz,"Sz",1)

El = LeftEnvironmentTensor(TensorMap(randn,ℂ^D1,ℂ^D1))
Er = RightEnvironmentTensor(TensorMap(randn,ℂ^D1,ℂ^D1))

# At = TensorMap(randn,ℂ^D1 ⊗ ℂ^d, ℂ^D2)
# A = MPSTensor(TensorMap(randn,ℂ^D1 ⊗ ℂ^d, ℂ^D2))
# B = MPSTensor(TensorMap(randn,ℂ^D1 ⊗ ℂ^d, ℂ^D2))

localto = TimerOutput()
@time for _ in 1:10
    # local tmp = similar(C.A,ComplexF64)
    # _action2_contract(C,El,op,op,Er)
    # @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * C.A[1,2,3,4] * op.A[-2,2] * op.A[-3,3] * Er.A[4,-4]
    # @time @tensor tmp[a,b;c,d] ≔ (((El.A[a,e] * C.A[e,f,g,h]) * op.A[b,f]) * op.A[c,g]) * Er.A[h,d]

    # let   
    # @tensor tmp1[-1,-2,-3;-4] ≔ El.A[-1,1] * C.A[1,2,-3,-4] * op.A[-2,2]
    # @tensor tmp2[-1,-2,-3;-4] ≔ tmp1[-1,-2,1,-4] * op.A[-3,1]
    # @tensor tmp3[-1,-2,-3;-4] ≔ tmp1[-1,-2,-3,1] * Er.A[1,-4]
    # end 


    # @time @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,3] * C.A[3,1,2,4] * op.A[-2,1] * op.A[-3,2] * Er.A[4,-4]
    EL1 = contract(El, C, op)
    EL2 = contract(EL1, op)
    tmp = contract(EL2, Er)
end



