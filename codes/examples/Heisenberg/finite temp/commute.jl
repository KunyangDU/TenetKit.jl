using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"
function ismergable(x::InteractionTreeLeave{N}, y::InteractionTreeLeave{M}) where {N,M}
    N ≠ M && return false
    N ≠ 1 && !isequal(x.A[1:end-1],y.A[1:end-1]) && return false
    
    
end


σ₊ = TensorMap([0 2;0 0],ℂ^2,ℂ^2)
σ₋ = TensorMap([0 0;2 0],ℂ^2,ℂ^2)
n = TensorMap([0 0;0 1],ℂ^2,ℂ^2)
Z = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)

x = InteractionTreeLeave(
    (σ₊, σ₋),
    (1,3),
    ("σ₊","σ₋"),
    (true,true),
    0.3,
    Z
)

y = InteractionTreeLeave(
    (σ₊, n),
    (2,4),
    ("σ₊","n"),
    (true,false),
    0.5,
    Z
)

z = InteractionTreeLeave(
    (σ₊, σ₋, σ₊, σ₋),
    (1,2,3,5),
    ("σ₊","σ₋","σ₊","σ₋"),
    (true,true,true,true),
    0.7,
    Z
)

# let x = x, y = y 
#     i = length(x)
#     j = 1
#     while j < length(y.)
    
# end


m = mul!(1,x,z)

show(x)
show(y)
show(m)



nothing