
TensorKit.inner(env₁::TaSKEnvironment{L,T},env₂::TaSKEnvironment{L,T}) where {L,T} = inner(env₁.TC, env₂.TC)
TensorKit.inner(A::Vector{T}, B::Vector{T}) where T <: Union{MPSTensor{3},DenseMPOTensor{4}} = sum(inner(A[i],B[i]) for i in 1:length(A))
TensorKit.norm(env::TaSKEnvironment) = sqrt(real(inner(env, env)))

function Base.:+(env₁::TaSKEnvironment{L,T}, env₂::TaSKEnvironment{L,T}) where {L,T}
    env = copy(env₁)
    for i in 1:L
        env.TC[i] = env₁.TC[i] + env₂.TC[i]
    end
    return env
end

function Base.:-(env₁::TaSKEnvironment{L,T}, env₂::TaSKEnvironment{L,T}) where {L,T}
    env = copy(env₁)
    for i in 1:L
        env.TC[i] = env₁.TC[i] - env₂.TC[i]
    end
    return env
end

Base.:*(α::Number, env::TaSKEnvironment{L}) where L = (e = copy(env); for i in 1:L; e.TC[i] = α * e.TC[i]; end; e)
Base.:*(env::TaSKEnvironment, α::Number) = α * env
Base.:/(env::TaSKEnvironment, α::Number) = (1/α) * env

normalize!(env::TaSKEnvironment) = (env.TC[:] = (1/norm(env)) .* env.TC; env)
