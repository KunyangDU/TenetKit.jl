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
include("Environment/contract.jl")

include("Hamiltonian/action.jl")
include("Hamiltonian/contract.jl")

include("IntrTree/addIntr.jl")
include("IntrTree/addIntr1.jl")
include("IntrTree/addIntr2.jl")
include("IntrTree/Automata.jl")
include("IntrTree/algebra.jl")

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
include("Algebra/contract.jl")

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
include("Algorithm/CBE_contract.jl")
include("Algorithm/DMRG_contract.jl")
include("Algorithm/TDVP_contract.jl")

include("LocalSpace/Fermion.jl")
include("LocalSpace/Spin.jl")
include("LocalSpace/trivial.jl")
