using MKL, TensorKit, JLD2, FiniteLattices, TimerOutputs, KrylovKit, AbstractTrees, SerializedElementArrays
import LinearAlgebra: BLAS, cross
import Statistics: std

include("Globals.jl")
include("init.jl")

include("TensorWrapper/AbstractType.jl")
include("Graph/node.jl")
include("Graph/edge.jl")
include("Graph/layermap.jl")
include("TensorWrapper/AbstractTensor.jl")
include("Environment/AbstractEnvironment.jl")
include("Interaction/operator.jl")
include("Observables/operator.jl")
include("Interaction/tunnel.jl")
include("Graph/dag.jl")
include("MPS/AbstractMPS.jl")
include("MPO/AbstractMPO.jl")
include("Hamiltonian/AbstractHamiltonian.jl")

# include("Observables/Node.jl")
# include("Observables/ObsTree.jl")
include("ProcessControl/AbstractType.jl")
include("ProcessControl/algorithm.jl")
include("ProcessControl/direction.jl")
include("ProcessControl/information.jl")
include("ProcessControl/scheme.jl")
include("ProcessControl/structure.jl")

include("Graph/optimize.jl")
include("Graph/iterator.jl")
include("Graph/operations.jl")

include("Defaults.jl")

include("TensorWrapper/TensorWrapper.jl")
include("TensorWrapper/canonicalize.jl")
include("TensorWrapper/method.jl")

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

include("Interaction/graph.jl")
include("Interaction/addIntr.jl")
include("Interaction/automata.jl")
include("Interaction/algebra.jl")

include("Observables/tunnel.jl")
include("Observables/graph.jl")
include("Observables/weight.jl")
include("Observables/addObs.jl")
include("Observables/update.jl")
include("Observables/tools.jl")
include("Observables/stack.jl")
include("Observables/calObs.jl")


include("tools/algebra.jl")
include("tools/geometry.jl")
include("tools/tools.jl")
include("tools/lattice.jl")
include("tools/vonNeumann.jl")
include("tools/swap.jl")

include("Algebra/inner.jl")
include("Algebra/mul1.jl")
include("Algebra/mul2.jl")
include("Algebra/axpby.jl")
include("Algebra/operations.jl")
# include("Algebra/densify.jl")
include("Algebra/contract.jl")

# include("utils/benchmarktools.jl")
include("utils/KrylovKit.jl")
include("utils/lattice.jl")
# include("utils/LKAN.jl")

include("Algorithm/DMRG.jl")
include("Algorithm/TDVP.jl")
include("Algorithm/SETTN.jl")
include("Algorithm/CBE.jl")
include("Algorithm/CBE2.jl")
include("Algorithm/CBE3-1.jl")
include("Algorithm/CBE3-3.jl")

include("Algorithm/CBE-SVD.jl")
include("Algorithm/orthogonalize.jl")
include("Algorithm/splice.jl")
include("Algorithm/utils.jl")
include("Algorithm/CBE_contract.jl")
include("Algorithm/DMRG_contract.jl")
include("Algorithm/TDVP_contract.jl")
include("Algorithm/XTRG.jl")

include("LocalSpace/Fermion.jl")
include("LocalSpace/Spin.jl")
include("LocalSpace/trivial.jl")
