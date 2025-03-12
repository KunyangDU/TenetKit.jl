
function Base.replace!(C::DenseMPO,A::DenseMPO)
    @assert C.L == A.L
    C.ts = A.ts
    C.center = A.center
    return C
end

"""
mul!(C, A, B, α, β) -> C
Combined inplace matrix-matrix or matrix-vector multiply-add \$A B α + C β\$.
The result is stored in C by overwriting it. Note that C must not be aliased with either A or B.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of C, A, B.
Nsweep: times of variational calculation (sweep). Default is 2.
"""
function mul!(C::DenseMPO, A::Union{DenseMPO,SparseMPO}, B::Union{DenseMPO,SparseMPO}, α::Number, β::Number; kwargs...)
    #D_MPO = get(kwargs, :D_MPO, maximum(vcat(map(size, filter(x -> typeof(x) <: DenseMPO.[A,B])[1].ts)...)))
    to = TimerOutput()
    D = get(kwargs, :D, maximum(vcat(collect.(map(size, filter(x -> typeof(x) <: DenseMPO, [A,B])[1].ts))...)))
    Nsweep = get(kwargs, :Nsweep, 10)
    tol = get(kwargs, :tol, 1e-12)
    @assert length(A) == length(B)
    L = length(A)

    tmp = deepcopy(C)'
    
    @timeit to "initialize ABC Env" begin
        EnvAB = Environment([deepcopy(A),deepcopy(B),tmp])
        EnvC = Environment([deepcopy(C),tmp])
        initialize!(EnvAB)
        initialize!(EnvC)
    end

    for i in 1:Nsweep
        localto = TimerOutput()
        ϵ = 0
        
        for site in 1:L-1
            x₀ = composite(EnvC.layer[1].ts[site:site+1]...)
            @timeit localto "contract" ts = map(z -> contract(z.envs[site], vcat(map(u -> z.layer[u].ts[site:site+1],1:length(z.layer)-1)...)..., z.envs[site+2]),[EnvAB,EnvC])
            @timeit localto "SVD" tl, tr, ~ = tsvd(axpby!(α, β, ts...); direction=:right,trunc = truncdim(D))
            @timeit localto "push right" map(z -> pushright!(z, tl, tr),[EnvAB,EnvC])
            x = composite(EnvC.layer[1].ts[site:site+1]...)
            ϵ = max(ϵ,norm(x-x₀))
        end
        for site in L:-1:2
            x₀ = composite(EnvC.layer[1].ts[site-1:site]...)
            @timeit localto "contract" ts = map(z -> contract(z.envs[site-1], vcat(map(u -> z.layer[u].ts[site-1:site],1:length(z.layer)-1)...)..., z.envs[site+1]),[EnvAB,EnvC])
            @timeit localto "SVD" tl, tr, ~ = tsvd(axpby!(α, β,ts...); direction=:left,trunc = truncdim(D))
            @timeit localto "push left" map(z -> pushleft!(z, tl, tr),[EnvAB,EnvC])
            x = composite(EnvC.layer[1].ts[site-1:site]...)
            ϵ = max(ϵ,norm(x-x₀))
        end
        # show(localto;title = "mul!")
        merge!(to,localto)

        ϵ < tol && break
    end

    @assert EnvAB.layer[end] == EnvC.layer[end]
    return xpy!(EnvAB.layer[end]',C),to
end

function mul!(C::DenseMPO, A::Union{DenseMPO,SparseMPO}, B::Union{DenseMPO,SparseMPO}; kwargs...)
    to = TimerOutput()
    D = get(kwargs, :D, maximum(vcat(collect.(map(size, filter(x -> typeof(x) <: DenseMPO, [A,B])[1].ts))...)))
    Nsweep = get(kwargs, :Nsweep, 10)
    tol = get(kwargs, :tol, 1e-12)
    @assert length(A) == length(B)
    L = length(A)

    tmp = deepcopy(C)'
    
    @timeit to "initialize ABC Env" begin
        EnvAB = Environment([deepcopy(A),deepcopy(B),tmp])
        initialize!(EnvAB)
    end

    for i in 1:Nsweep
        localto = TimerOutput()
        ϵ = 0
        
        for site in 1:L-1
            x₀ = composite(EnvAB.layer[1].ts[site:site+1]...)
            @timeit localto "contract" ts = contract(EnvAB.envs[site], vcat(map(u -> EnvAB.layer[u].ts[site:site+1],1:length(EnvAB.layer)-1)...)..., EnvAB.envs[site+2])
            @timeit localto "SVD" tl, tr, ~ = tsvd(ts; direction=:right,trunc = truncdim(D))
            @timeit localto "push right" pushright!(EnvAB, tl, tr)
            x = composite(EnvAB.layer[1].ts[site:site+1]...)
            ϵ = max(ϵ,norm(x-x₀))
        end
        for site in L:-1:2
            x₀ = composite(EnvAB.layer[1].ts[site-1:site]...)
            @timeit localto "contract" ts = contract(EnvAB.envs[site-1], vcat(map(u -> EnvAB.layer[u].ts[site-1:site],1:length(EnvAB.layer)-1)...)..., EnvAB.envs[site+1])
            @timeit localto "SVD" tl, tr, ~ = tsvd(ts; direction=:left,trunc = truncdim(D))
            @timeit localto "push left" pushleft!(EnvAB, tl, tr)
            x = composite(EnvAB.layer[1].ts[site-1:site]...)
            ϵ = max(ϵ,norm(x-x₀))
        end
        # show(localto;title = "mul!")
        merge!(to,localto)

        ϵ < tol && break
    end

    return xpy!(EnvAB.layer[end]',C),to
end
"""
axpy!(α, x, y) -> y
Overwrite y with x * α + y and return y. If x and y have the same axes, it's equivalent with y .+= x .* a.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of x, y.
Nsweep: times of variational calculation (sweep). Default is 2.

"""
function axpy!(α::Number, x::DenseMPO{L}, y::DenseMPO{L};kwargs...) where L
    D = get(kwargs, :D, max(map(y -> maximum(vcat(collect.(map(size, y.ts))...)),[x,y])...))
    N = get(kwargs, :N, 10)
    tol = get(kwargs, :tol, 1e-12)
    tmp = deepcopy(y)'
    
    to = TimerOutput()
    @timeit to "initialize XY Env" begin
        Envx = Environment([deepcopy(x),tmp])
        Envy = Environment([deepcopy(y),tmp])
        initialize!(Envx)
        initialize!(Envy)
    end

    for i in 1:N
        ϵ = 0
        localto = TimerOutput()
        for site in 1:L-1
            x₀ = composite(Envx.layer[1].ts[site:site+1]...)
            @timeit localto "contract" ts = map(z -> contract(z.envs[site], z.layer[1].ts[site:site+1]..., z.envs[site+2]),[Envx,Envy])
            @timeit localto "SVD" tl, tr, temptruncerr = tsvd(axpy!(α,ts...); direction=:right,trunc = truncdim(D))
            @timeit localto "push right" map(z -> pushright!(z, tl, tr),[Envx,Envy])
            x = composite(Envx.layer[1].ts[site:site+1]...)
            ϵ = max(ϵ,norm(x-x₀))
        end
        for site in L:-1:2
            x₀ = composite(Envx.layer[1].ts[site-1:site]...)
            @timeit localto "contract" ts = map(z -> contract(z.envs[site-1], z.layer[1].ts[site-1:site]..., z.envs[site+1]),[Envx,Envy])
            @timeit localto "SVD" tl, tr, temptruncerr = tsvd(axpy!(α,ts...); direction=:left,trunc = truncdim(D))
            @timeit localto "push left" map(z -> pushleft!(z, tl, tr),[Envx,Envy])
            x = composite(Envx.layer[1].ts[site-1:site]...)
            ϵ = max(ϵ,norm(x-x₀))
        end

        # show(localto;title = "axpy!")
        merge!(to,localto)

        ϵ < tol && break
    end

    @assert Envx.layer[2] == Envy.layer[2]
    return xpy!(Envx.layer[2]',y),to
end

function axpy!(α::Number, x::CompositeMPOTensor{N₁,R₁}, y::CompositeMPOTensor{N₂,R₂}) where {N₁,R₁,N₂,R₂}
    return axpby!(α,1,x,y)
end

function axpby!(α::Number, β::Number, x::CompositeMPOTensor{N₁,R₁}, y::CompositeMPOTensor{N₂,R₂}) where {N₁,R₁,N₂,R₂}
    @assert N₁ == N₂ && R₁ == R₂
    y.A = x.A * α + y.A * β
    return y
end

function axpby!(::Number, β::Number, ::Nothing, y::CompositeMPOTensor)
    y.A = y.A * β
    return y
end

function xpy!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO}
    y.ts[:] = x.ts[:]
    y.center = x.center
    return y
end

function tr(ρ::DenseMPO)
    return tr(ρ,ρ')
end

function tr(ρ1::DenseMPO,ρ2::AdjointMPO)
    Env = Environment([deepcopy(ρ1),deepcopy(ρ2)])
    initialize!(Env)
    return _scalar(Env)
end

function tr(ρ::DenseMPO, Opr::SparseMPO)
    Env = Environment([deepcopy(ρ), Opr, ρ'])
    initialize!(Env)
    return _scalar(Env)
end

"""
compatible for N-layer Environment
"""
function _scalar(Env::Environment{N}) where N
    @assert (site = Env.center[1]) == Env.center[2]
    t1 = map(x -> Env.layer[x].ts[site], 1:length(Env.layer))
    tmp = contract(Env.envs[site],t1...,Env.envs[site+1])
    return tmp
end

function scalar(Env::Environment{3})
    @assert Env.center[1] == Env.center[2]
    contract(Env.layer[3].ts[Env.center[1]], action(proj1(Env, Env.center[1]), Env.layer[1].ts[Env.center[1]]))
end



