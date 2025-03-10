abstract type AbstractInformation end

abstract type AlgorithmInfo <: AbstractInformation end
abstract type SolverInfo <: AbstractInformation end

abstract type AbstractDirection end
abstract type SweepDirection <: AbstractDirection end


abstract type AbstractAlgorithm end
abstract type SolverAlgo <: AbstractAlgorithm end


abstract type AbstractScheme end
abstract type SweepScheme <: AbstractScheme end
abstract type CBEscheme <: AbstractScheme end





