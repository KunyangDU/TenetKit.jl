using MKL, TensorKit, JLD2, FiniteLattices, BenchmarkTools, TimerOutputs, KrylovKit, AbstractTrees
import LinearAlgebra: BLAS 

include("Globals.jl")
include("init.jl")

include("TensorWrapper/AbstractType.jl")
include("TensorWrapper/AbstractTensor.jl")
include("MPS/AbstractMPS.jl")
include("MPO/AbstractMPO.jl")
include("Environment/AbstractEnvironment.jl")
include("Hamiltonian/AbstractHamiltonian.jl")
include("IntrTree/LocalOperator.jl")
include("IntrTree/Node.jl")
include("Observables/Node.jl")
include("Observables/ObsTree.jl")
include("ProcessControl/AbstractType.jl")
include("ProcessControl/algorithm.jl")
include("ProcessControl/direction.jl")
include("ProcessControl/information.jl")
include("ProcessControl/scheme.jl")
include("ProcessControl/structure.jl")

include("Defaults.jl")

include("TensorWrapper/TensorWrapper.jl")
include("TensorWrapper/canonicalize.jl")

include("MPS/contract.jl")
include("MPS/methods.jl")
include("MPS/operations.jl")

include("MPO/contract.jl")
include("MPO/methods.jl")
include("MPO/operations.jl")

include("Environment/operations.jl")
include("Environment/push.jl")

include("Hamiltonian/action.jl")

include("IntrTree/addIntr.jl")
include("IntrTree/addIntr1.jl")
include("IntrTree/addIntr2.jl")
include("IntrTree/Automata.jl")

include("Observables/addObs.jl")
include("Observables/calObs.jl")

include("tools/algebra.jl")
include("tools/geometry.jl")
include("tools/tools.jl")
include("tools/lattice.jl")

include("Algebra/inner.jl")
include("Algebra/mul1.jl")
include("Algebra/mul2.jl")
include("Algebra/axpby.jl")
include("Algebra/operations.jl")
include("Algebra/densify.jl")

include("utils/benchmarktools.jl")
include("utils/KrylovKit.jl")
include("utils/lattice.jl")

include("Algorithm/DMRG.jl")
include("Algorithm/TDVP.jl")
include("Algorithm/SETTN.jl")
include("Algorithm/CBE.jl")
include("Algorithm/CBE2.jl")
include("Algorithm/CBE3.jl")
include("Algorithm/CBE-SVD.jl")
include("Algorithm/orthogonalize.jl")
include("Algorithm/splice.jl")
include("Algorithm/utils.jl")

include("LocalSpace/Fermion.jl")
include("LocalSpace/Spin.jl")
include("LocalSpace/trivial.jl")

# include("default.jl")
# include("TempSrc/AbstractType.jl")

# include("Environment/Environment.jl")
# include("Environment/Initialize.jl")
# include("Environment/Pushleft.jl")
# include("Environment/Pushright.jl")

# include("Operations/Move.jl")
# include("Operations/Merge.jl")
# include("Operations/Contract.jl")
# include("Operations/SVD.jl")
# include("Operations/Variation.jl")

# include("MPS/MPS.jl")
# include("MPS/Operations.jl")

# include("MPO/MPO.jl")
# include("MPO/Normalize.jl")
# include("MPO/Operations.jl")
# include("MPO/Operators.jl")
# include("MPO/ObsMPO.jl")

# include("tools/tools.jl")
# include("tools/geometry.jl")
# include("tools/algebra.jl")

# include("IntrTree/LocalOperator.jl")
# include("IntrTree/Node.jl")
# include("IntrTree/addIntr.jl")
# include("IntrTree/addIntr1.jl")
# include("IntrTree/addIntr2.jl")
# include("IntrTree/Automata.jl")

# include("Observables/ObsTree.jl")
# include("Observables/calObs.jl")
# include("Observables/addObs.jl")
# include("Observables/Observables.jl")

# include("LocalSpace/Fermion.jl")
# include("LocalSpace/Spin.jl")

# include("Algorithm/DMRG.jl")
# include("Algorithm/Lanczos.jl")
# include("Algorithm/TDVP.jl")
# include("Algorithm/SETTN.jl")
# include("Algorithm/tanTRG.jl")




# include("TempSrc/AbstractTensor.jl")
# include("TempSrc/AbstractMPS.jl")
# include("TempSrc/AbstractMPO.jl")
# include("TempSrc/AbstractEnvironment.jl")
# include("TempSrc/utils.jl")
# include("TempSrc/SymSpin.jl")
# include("TempSrc/SymFermion.jl")
# include("TempSrc/canonicalize.jl")
# include("TempSrc/push.jl")
# include("TempSrc/AbstractHamiltonian.jl")
# include("TempSrc/action.jl")
# include("TempSrc/contract_MPO.jl")
# include("TempSrc/contract_MPS.jl")
# include("TempSrc/Lanczos.jl")
# include("TempSrc/DMRG.jl")
# include("TempSrc/TDVP.jl")
# include("TempSrc/observables.jl")
# include("TempSrc/operations.jl")
# include("TempSrc/SETTN.jl")
# include("TempSrc/benchmarktools.jl")
# include("TempSrc/CBE.jl")

# include("TempSrc/TensorWrapper.jl")

#= 
MPO data matrix should be the hermitian conjugate of the 
matrix of observables, i.e.
M_{data} = M_{Obs}⁺
under this convention, the evolution operator is exp(i)
=#

#= 
Suppose you have been familiar with TensorKit.jl
Indexing formalism of TensorMap
[] -> codomain, () -> domain

I. MPS
    1. Right Orthogonal:
           ___
    (2) → |   | → [1]
           ‾‾‾
            ↑
           (3)
    i.e. TensorMap{ [1] , (2) ⊗ (3) }
           [2] 
            ↑
           ___
    [1] ← |   | ← (3)
           ‾‾‾

    i.e. TensorMap{ [1] ⊗ [2], (3)  }
    -------------------
    2. Left Orthogonal:
           ___
    [1] ← |   | ← (3)
           ‾‾‾
            ↑
           (2)
    i.e. TensorMap{ [1] , (2) ⊗ (3) }
           [1]
            ↑
           ___
    (3) → |   | → [2]
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2]  , (3) }
    -------------------
    3. Center:
           ___
    (1) → |   | ← (3)
           ‾‾‾
            ↑
           (2)
    i.e. TensorMap{ (1) ⊗ (2) ⊗ (3)}
           [2]
            ↑
           ___
    [1] ← |   | → [3]
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3]}
    -------------------
    4. SVD conjunction
    -Right
           ___
    (2) → | → | ← (1)
           ‾‾‾
    i.e. TensorMap{ (1) ⊗ (2) }

    -Left
           ___
    (1) → | ← | ← (2)
           ‾‾‾
    i.e. TensorMap{ (1) ⊗ (2) }
    -------------------
    5. 2-site Center MPS
           _______
    (1) → |       | ← (4)
           ‾‾‾‾‾‾‾
            ↑   ↑
           (2) (3)
    i.e. TensorMap{ (1) ⊗ (2) ⊗ (3) ⊗ (4) }
           [2] [3]
            ↑   ↑
           _______
    [1] ← |       | → [4]
           ‾‾‾‾‾‾‾
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3] ⊗ [4] }

II.MPO
    1.1 Local MPO (Right Orthogonal)
           [2]
            ↑
           ___
    (3) → |   | → [1]
           ‾‾‾
            ↑
           (4)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

           (4)
            ↓
           ___
    [1] ← |   | ← (3)
           ‾‾‾
            ↓
           [2]
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

    1.2 Local MPO (Left Orthogonal)
           [1]
            ↑
           ___
    [2] ← |   | ← (4)
           ‾‾‾
            ↑
           (3)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

           (3)
            ↓
           ___
    (4) → |   | → [2]
           ‾‾‾
            ↓
           [1]
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

    1.3 Local MPO (Center Orthogonal)
           [1]
            ↑
           ___
    (2) → |   | ← (4)
           ‾‾‾
            ↑
           (3)
    i.e. TensorMap{ [1], (2) ⊗ (3) ⊗ (4) }

           (4)
            ↓
           ___
    [1] ← |   | → [3]
           ‾‾‾
            ↓
           [2]
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3], (4) }

    1.4 Local 2 site MPO (Center Orthogonal)
           [2] [1]
            ↑   ↑
           _______
    (3) → |       | ← (6)
           ‾‾‾‾‾‾‾
            ↑   ↑
           (4) (5)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) ⊗ (5) ⊗ (6) }

           (4)
            ↓
           ___
    [1] ← |   | → [3]
           ‾‾‾
            ↓
           [2]
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3], (4) }

    2. 0-site Effective Hamiltonian
         [1] [2]
          ↑   ↑
         _______
        |       |
         ‾‾‾‾‾‾‾
          ↑   ↑
         (3) (4)
    i.e. TensorMap{ [1] ⊗ [2], (3) ⊗ (4) }

    3. 1-site Effective Hamiltonian
         [1] [2] [3]
          ↑   ↑   ↑
         ___________
        |           |
         ‾‾‾‾‾‾‾‾‾‾‾
          ↑   ↑   ↑
         (4) (5) (6)
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3], (4) ⊗ (5) ⊗ (6) }

    4. 2-site Effective Hamiltonian
         [1] [2] [3] [4]
          ↑   ↑   ↑   ↑
         _______________
        |               |
         ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
          ↑   ↑   ↑   ↑
         (5) (6) (7) (8)
    i.e. TensorMap{ [1] ⊗ [2] ⊗ [3] ⊗ [4], (5) ⊗ (6) ⊗ (7) ⊗ (8) }

III.Environment
    1. Right Environment
           ___
    [1] ← |   |
    (2) → |   |
    (3) → |   |
           ‾‾‾
    i.e. TensorMap{ [1] , (2) ⊗ (3) }

    *canonicalized
           ___
    [1] ← |   |
    [2] ← |   |
    (3) → |   |
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2], (3) }

           ___
    [1] ← |   |
    (2) → |   |
           ‾‾‾
    i.e. TensorMap{ [1] , (2) }

    2. Left Environment
           ___
          |   | → [1]
          |   | → [2]
          |   | ← (3)
           ‾‾‾
    i.e. TensorMap{ [1] ⊗ [2], (3)  }
           ___
          |   | → [1]
          |   | ← (2)
           ‾‾‾
    i.e. TensorMap{ [1] , (2)  }

 =#



