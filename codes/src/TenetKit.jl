using MKL, TensorKit, JLD2, FiniteLattices, TimerOutputs, KrylovKit, SerializedElementArrays
import LinearAlgebra: BLAS, cross, diagm
import Statistics: std

include("Globals.jl")
include("init.jl")

include("tools/TensorKit.jl")

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

include("ProcessControl/truncations.jl")
include("ProcessControl/AbstractType.jl")
include("ProcessControl/algorithm.jl")
include("ProcessControl/direction.jl")
include("ProcessControl/information.jl")
include("ProcessControl/scheme.jl")
include("ProcessControl/structure.jl")

# include("Graph/optimize.jl")
include("Graph/bipartite.jl")
include("Graph/iterator.jl")
include("Graph/myhill.jl")
# include("Graph/operations.jl")

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

include("Observables/weight.jl")
include("Observables/graph.jl")
include("Observables/addObs.jl")
include("Observables/update.jl")
include("Observables/tools.jl")
include("Observables/stack.jl")
include("Observables/calObs.jl")
include("Observables/instance.jl")

include("tools/algebra.jl")
include("tools/geometry.jl")
include("tools/tools.jl")
include("tools/lattice.jl")
include("tools/vonNeumann.jl")
include("tools/swap.jl")
include("tools/KrylovKit.jl")

include("Algebra/inner.jl")
include("Algebra/mul.jl")
include("Algebra/axpby.jl")
include("Algebra/operations.jl")
include("Algebra/contract.jl")


include("Algorithm/DMRG/DMRG.jl")
include("Algorithm/DMRG/contract.jl")
include("Algorithm/TDVP/TDVP.jl")
include("Algorithm/TDVP/contract.jl")
include("Algorithm/CBE/CBE.jl")
include("Algorithm/CBE/CBE2.jl")
include("Algorithm/CBE/CBE3-1.jl")
include("Algorithm/CBE/CBE3-3.jl")
include("Algorithm/CBE/svd.jl")
include("Algorithm/CBE/orthogonalize.jl")
include("Algorithm/CBE/splice.jl")
include("Algorithm/CBE/contract.jl")
include("Algorithm/XTRG.jl")
include("Algorithm/SETTN.jl")
include("Algorithm/utils.jl")

include("LocalSpace/Fermion/symmetric.jl")
include("LocalSpace/Fermion/trivial.jl")
include("LocalSpace/Spin/symmetric.jl")
include("LocalSpace/Spin/trivial.jl")

