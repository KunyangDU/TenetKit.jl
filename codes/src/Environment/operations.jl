
function initialize!(env::Environment;kwargs...)
    envs_vec = Vector{AbstractEnvironmentTensor}(undef, env.L + 1)
    if env.isdisk
        env.envs = _disk(envs_vec)
    else
        env.envs = envs_vec
    end
    setdefault!(env,env.layer...;kwargs...)
    canonicalize!(env,1)
end

function canonicalize!(env::Environment,sl::Int64,sr::Int64)
    @assert 1 ≤ sl ≤ sr ≤ env.L + 1

    for _ in env.center[1]:sl-1
        pushright!(env)
    end

    for _ in env.center[2]:-1:sr+1
        pushleft!(env)
    end

end

function canonicalize!(env::Environment,si::Int64)
    @assert 1 ≤ si ≤ env.L + 1
    canonicalize!(env,si,si)
end

function setdefault!(env::Environment{3},A::T₁,H::SparseMPO{L},B::T₂;kwargs...) where {T₁ <: Union{DenseMPS{L},DenseMPO{L}}, T₂ <: Union{AdjointMPS{L},AdjointMPO{L},RefMPS{L},RefMPO{L}}} where L
    als, ars = auxspace(A,H,B)
    env.envs[1] = SparseLeftEnvironmentTensor(ones(als[3], als[2] ⊗ als[1]))
    env.envs[end] = SparseRightEnvironmentTensor(ones(ars[1] ⊗ ars[2], ars[3]))
end
function setdefault!(env::Environment{2},A::T₁,B::T₂;kwargs...)where {T₁ <: Union{DenseMPS{L},DenseMPO{L}}, T₂ <: Union{AdjointMPS{L},AdjointMPO{L},RefMPS{L},RefMPO{L}}} where L
    als, ars = auxspace(A,B)
    env.envs[1] = DenseLeftEnvironmentTensor(ones(als[2], als[1]))
    env.envs[end] = DenseRightEnvironmentTensor(ones(ars[1], ars[2]))
end

function setdefault!(env::Environment{3},A::T₁,H::T₂,B::T₃;kwargs...) where {T₁ <: Union{DenseMPS{L},DenseMPO{L}}, T₂ <: Union{RefMPS{L},RefMPO{L}}, T₃ <: Union{AdjointMPS{L},AdjointMPO{L},RefMPS{L},RefMPO{L}}} where L
    als, ars = auxspace(A,H,B)
    env.envs[1] = DenseLeftEnvironmentTensor(ones(als[1] ⊗ als[2], als[3]))
    env.envs[end] = DenseRightEnvironmentTensor(ones(ars[1], ars[2] ⊗ ars[3]))
end

auxspace(A::T) where T <: Union{DenseMPS,DenseMPO,AdjointMPS,AdjointMPO,RefMPS,RefMPO} = left_auxspace(A[1]),right_auxspace(A[end])
auxspace(A::InteractionTunnel) = left_auxspace(A.A[1]),right_auxspace(A.A[end])
auxspace(A::SparseMPO) = A.auxspace

left_auxspace(A::MPSTensor{3}) = space(A,1)
left_auxspace(A::AdjointMPSTensor{3}) = space(A,2)'
left_auxspace(A::DenseMPOTensor{4}) = space(A,2)
left_auxspace(A::AdjointMPOTensor{4}) = space(A,4)'

right_auxspace(A::MPSTensor{3}) = space(A,3)'
right_auxspace(A::AdjointMPSTensor{3}) = space(A,1)
right_auxspace(A::DenseMPOTensor{4}) = space(A,3)'
right_auxspace(A::AdjointMPOTensor{4}) = space(A,1)

left_auxspace(::T) where T <: Union{LocalOperator{1,1},LocalOperator{1,2},IdentityOperator} = nothing
left_auxspace(A::T) where T <: Union{LocalOperator{2,1},LocalOperator{2,2}} = space(A,2)
right_auxspace(::T) where T <: Union{LocalOperator{1,1},LocalOperator{2,1},IdentityOperator} = nothing
right_auxspace(A::LocalOperator{1,2}) = space(A,2)'
right_auxspace(A::LocalOperator{2,2}) = space(A,3)'

TensorKit.:⊗(::Nothing, A::T) where T <: Union{ElementarySpace,ProductSpace} = A
TensorKit.:⊗(A::T, ::Nothing) where T <: Union{ElementarySpace,ProductSpace} = A
auxspace(a, b, args...) = begin
    xs = map(auxspace, (a, b, args...))
    return (map(first, xs), map(last, xs))
end

# function setdefault!(env::Environment{3};kwargs...)
#     if issparse(env.layer[2])
#         lds = get(kwargs,:left_default_space, reverse(map(x -> getAuxSpace(env.layer[x][1])[1],[1,3])))
#         rds = get(kwargs,:right_default_space, map(x -> getAuxSpace(env.layer[x][end])[2],[1,3]))
#         env.envs[1] = SparseLeftEnvironmentTensor(isometry(lds...))
#         env.envs[end] = SparseRightEnvironmentTensor(isometry(rds...))
#     elseif isadjoint(env.layer[2]) | (isref(env.layer[2]) && env.layer[2].mapping == adjoint)
#         AuxSpaces = reverse(map(x -> getAuxSpace(env.layer[x][1])[1],1:3))
#         env.envs[1] = DenseLeftEnvironmentTensor(isometry(AuxSpaces[1] ⊗ AuxSpaces[2],AuxSpaces[3]))
#         AuxSpaces = map(x -> getAuxSpace(env.layer[x][end])[2],1:3)
#         env.envs[end] = DenseRightEnvironmentTensor(isometry(AuxSpaces[1],  AuxSpaces[2] ⊗ AuxSpaces[3]))
#     else
#         AuxSpaces = reverse(map(x -> getAuxSpace(env.layer[x][1])[1],1:3))
#         env.envs[1] = DenseLeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2] ⊗ AuxSpaces[3]))
#         AuxSpaces = map(x -> getAuxSpace(env.layer[x][end])[2],1:3)
#         env.envs[end] = DenseRightEnvironmentTensor(isometry(AuxSpaces[1] ⊗ AuxSpaces[2], AuxSpaces[3]))
#     end
# end

# function setdefault!(env::Environment{2};kwargs...)
#     if !issparse(env.layer[1]) && !issparse(env.layer[2])
#         env.envs[1] = DenseLeftEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x][1])[1],1:2)...))
#         env.envs[end] = DenseRightEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x][end])[2],1:2)...))
#     elseif issparse(env.layer[1]) && !issparse(env.layer[2])
#         lds = get(kwargs,:left_default_space, getAuxSpace(env.layer[2][1])[1] |> y -> (trivial(y),y))
#         rds = get(kwargs,:right_default_space, getAuxSpace(env.layer[2][end])[2] |> y -> (y,trivial(y)))
#         env.envs[1] = SparseLeftEnvironmentTensor(isometry(lds...))
#         env.envs[end] = SparseRightEnvironmentTensor(isometry(rds...))
#     elseif !issparse(env.layer[1]) && issparse(env.layer[2])
#         lds = get(kwargs,:left_default_space, repeat([getAuxSpace(env.layer[1][1])[1],],2))
#         rds = get(kwargs,:right_default_space, repeat([getAuxSpace(env.layer[1][end])[2],],2))
#         env.envs[1] = SparseLeftEnvironmentTensor(isometry(lds...))
#         env.envs[end] = SparseRightEnvironmentTensor(isometry(rds...))
#     end
# end

# function setdefault!(Env::Environment{4})
#     if issparse(Env.layer[1]) && issparse(Env.layer[3])
#         ρ1 = Env.layer[2]
#         ρ2′ = Env.layer[4]
#         tmpL = (isometry(space(ρ2′[1])[4]', space(ρ1[1])[2]))
#         tmpR = (isometry(space(ρ1[end])[3]', space(ρ2′[end])[1]))
#         Env.envs[1] = SparseLeftEnvironmentTensor([tmpL;;])
#         Env.envs[end] = SparseRightEnvironmentTensor([tmpR;;])
#     end
# end

