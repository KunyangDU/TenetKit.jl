using TensorKit

include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^10
LocalSpace = TrivialSpinOneHalf
d = LocalSpace.PhySpace
C = CompositeMPOTensor(TensorMap(randn,d⊗d⊗ℂ^D,ℂ^D⊗d⊗d))
op = LocalOperator(LocalSpace.Sz, "Sz", 1)
El = LeftEnvironmentTensor(TensorMap(randn,ℂ^D,ℂ^D))
Er = RightEnvironmentTensor(TensorMap(randn,ℂ^D,ℂ^D))
localto = TimerOutput()
@time for _ in 1:10
    @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-3,1] * C.A[3,2,1,4,-5,-6] * op.A[-2,2] * op.A[-1,3] * Er.A[4,-4]
    # @timeit localto "_action2_EL1=El_obj_H1" EL1 = contract(El, C, op)
    # @timeit localto "_action2_EL2=EL1_H2" EL2 = contract(EL1, op)
    # @timeit localto "_action2_C=EL2_Er" tmp = contract(EL2, Er)
    # @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * C.A[1,2,3,4] * op.A[-2,2] * op.A[-3,3] * Er.A[4,-4]
end
# localto
