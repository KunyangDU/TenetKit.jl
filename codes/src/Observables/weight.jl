mutable struct ObservableWeight
    EnvL::Union{LeftEnvironmentTensor,Nothing}
    EnvR::Union{RightEnvironmentTensor,Nothing}
    leftdata::Union{Dict,Nothing}
    rightdata::Union{Dict,Nothing}
    ObservableWeight() = new(nothing,nothing,nothing,nothing)
end

isdefault(A::DirectedEdge{ObservableWeight}) = isnothing(A.weight.EnvL) && isnothing(A.weight.EnvR) && isnothing(A.weight.leftdata) && isnothing(A.weight.rightdata)
inherit_weight!(A::DirectedEdge{ObservableWeight}, w::ObservableWeight) = nothing
default_weight!(A::DirectedEdge{ObservableWeight}) = default_weight!(A.weight)
function default_weight!(A::ObservableWeight)
    A.EnvL = nothing
    A.EnvR = nothing
end