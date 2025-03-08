abstract type AbstractTensorWrapper end

abstract type AbstractMPSTensor{R} <: AbstractTensorWrapper end
abstract type AbstractMPOTensor <: AbstractTensorWrapper end
abstract type AbstractEnvironmentTensor <: AbstractTensorWrapper end

abstract type AbstractLeftEnvironmentTensor <: AbstractEnvironmentTensor end
abstract type AbstractRightEnvironmentTensor <: AbstractEnvironmentTensor end

abstract type AbstractMPWrapper end

abstract type AbstractMPS <: AbstractMPWrapper end
abstract type AbstractMPO <: AbstractMPWrapper end

abstract type AbstractEnvironment end

abstract type AbstractHamiltonian end
abstract type AbstractProjectiveHamiltonian <: AbstractHamiltonian end

abstract type AbstractObservable end
abstract type AbstractLocalOperator end

