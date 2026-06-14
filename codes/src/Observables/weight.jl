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
function default_weight!(A::ObservableWeight,isstrength::Bool = true)
    A.EnvL = nothing
    A.EnvR = nothing
    A.leftdata = nothing
    A.rightdata = nothing
    isstrength && (A.strength = 1.0)
end