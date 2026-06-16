mutable struct ObservableWeight
    EnvL::Union{LeftEnvironmentTensor,Nothing}
    EnvR::Union{RightEnvironmentTensor,Nothing}
    leftdata::Union{Dict,Nothing}
    rightdata::Union{Dict,Nothing}
    strength::Number
    lock::ReentrantLock
    ObservableWeight(strength::Number = 1.0) = new(nothing,nothing,nothing,nothing,strength,ReentrantLock())
end

composite(A::ObservableWeight, B::ObservableWeight) = ObservableWeight(A.strength * B.strength)

isdefault(A::DirectedEdge{ObservableWeight}) = isnothing(A.weight.EnvL) && isnothing(A.weight.EnvR) && isnothing(A.weight.leftdata) && isnothing(A.weight.rightdata) && (A.weight.strength == 1)
isleftdefault(A::DirectedEdge{ObservableWeight}) = isnothing(A.weight.EnvL) && isnothing(A.weight.leftdata)
isrightdefault(A::DirectedEdge{ObservableWeight}) = isnothing(A.weight.EnvR) && isnothing(A.weight.rightdata)

inherit_weight!(A::DirectedEdge{ObservableWeight}, w::ObservableWeight) = (A.weight.strength *= w.strength)
default_weight!(A::DirectedEdge{ObservableWeight}, isstrength::Bool = true) = default_weight!(A.weight, isstrength)
leftdelfault_weight!(A::DirectedEdge{ObservableWeight}) = leftdelfault_weight!(A.weight)
rightdelfault_weight!(A::DirectedEdge{ObservableWeight}) = rightdelfault_weight!(A.weight)
function default_weight!(A::ObservableWeight,isstrength::Bool = true)
    leftdelfault_weight!(A)
    rightdelfault_weight!(A)
    isstrength && (A.strength = 1.0)
end
function leftdelfault_weight!(A::ObservableWeight)
    A.EnvL = nothing
    A.leftdata = nothing
end
function rightdelfault_weight!(A::ObservableWeight)
    A.EnvR = nothing
    A.rightdata = nothing
end

