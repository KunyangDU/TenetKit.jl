
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
function mul!(C::DenseMPO, A::Union{DenseMPO{L₁},SparseMPO{L₁}}, B::Union{DenseMPO{L₂},SparseMPO{L₂}}, α::Number, β::Number, Alg::Algebraalgo; kwargs...) where {L₁,L₂}

    @assert L₁ == L₂
    to = TimerOutput()

    tmp = deepcopy(C)'
    
    @timeit to "initialize ABC Env" begin
        EnvAB = Environment([deepcopy(A),deepcopy(B),tmp])
        EnvC = Environment([deepcopy(C),tmp])
        initialize!(EnvAB)
        initialize!(EnvC)
    end
    info = Algebrainfo()
    while info.n ≤ Alg.N
        localto = TimerOutput()

        l2rinfo = Algebrasweepinfo(L2R())
        mto = mul!(EnvC,EnvAB,α,β,Alg,l2rinfo)
        merge!(localto,mto)
        merge!(info,l2rinfo)

        r2linfo = Algebrasweepinfo(R2L())
        mto = mul!(EnvC,EnvAB,α,β,Alg,r2linfo)
        merge!(localto,mto)
        merge!(info,r2linfo)

        show(localto;title = "mul!")
        print("\n")
        show(info)
        merge!(to,localto)

        info.err < Alg.tol && break
    end

    @assert EnvAB.layer[end] == EnvC.layer[end]
    return xp!(EnvAB.layer[end]',C),to,info
end

function mul!(EnvC::Environment{2}, EnvAB::Environment{3}, α::Number, β::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R}; kwargs...)
    localto = TimerOutput()
    L = length(EnvC.layer[1])
    for site in 1:L-1
        localinfo = Algebrasiteinfo()
        x₀ = composite(EnvC.layer[1].ts[site:site+1]...)
        @timeit localto "composite" ts = map(z -> contract(z.envs[site], vcat(map(u -> z.layer[u].ts[site:site+1],1:length(z.layer)-1)...)..., z.envs[site+2]),[EnvAB,EnvC])
        @timeit localto "SVD" tl, tc, tr, ~ = tsvd(axpby!(α, β, ts...); direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tr = contract(tc,tr) 
        # @timeit localto "SVD" tl, tr, ~ = tsvd(axpby!(α, β, ts...); direction=:right,trunc = Alg.trunc)
        @timeit localto "push right" map(z -> pushright!(z, tl, tr),[EnvAB,EnvC])
        x = composite(EnvC.layer[1].ts[site:site+1]...)
        localinfo.err = norm(x-x₀)
        merge!(sweepinfo,localinfo)
    end

    return localto
end

function mul!(EnvC::Environment{2}, EnvAB::Environment{3}, α::Number, β::Number, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L}; kwargs...)
    localto = TimerOutput()
    L = length(EnvC.layer[1])
    for site in L:-1:2
        localinfo = Algebrasiteinfo()
        x₀ = composite(EnvC.layer[1].ts[site-1:site]...)
        @timeit localto "composite" ts = map(z -> contract(z.envs[site-1], vcat(map(u -> z.layer[u].ts[site-1:site],1:length(z.layer)-1)...)..., z.envs[site+1]),[EnvAB,EnvC])
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(axpby!(α, β, ts...); direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tl = contract(tl,tc) 
        # @timeit localto "SVD" tl, tr, ~ = tsvd(axpby!(α, β,ts...); direction=:left,trunc = Alg.trunc)
        @timeit localto "push left" map(z -> pushleft!(z, tl, tr),[EnvAB,EnvC])
        x = composite(EnvC.layer[1].ts[site-1:site]...)
        localinfo.err = norm(x-x₀)
        merge!(sweepinfo,localinfo)
    end

    return localto
end

function mul!(C::DenseMPO, A::Union{DenseMPO,SparseMPO}, B::Union{DenseMPO,SparseMPO}; kwargs...)
    trunc = get(kwargs,:trunc,notrunc())
    N  = get(kwargs,:N,3)
    tol = get(kwargs,:tol,1e-12)
    algo = Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,N,tol)
    return mul!(C,A,B,1,1,algo;kwargs...)
end

"""
axpy!(α, x, y) -> y
Overwrite y with x * α + y and return y. If x and y have the same axes, it's equivalent with y .+= x .* a.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of x, y.
Nsweep: times of variational calculation (sweep). Default is 2.

"""
function axpby!(α::Number, β::Number, x::DenseMPO{L}, y::DenseMPO{L};kwargs...) where L
    trunc = get(kwargs,:trunc,notrunc())
    N  = get(kwargs,:N,3)
    tol = get(kwargs,:tol,1e-12)
    algo = Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,N,tol)
    return axpby!(α,β,x,y,algo;kwargs...)
end

function axpby!(α::Number, β::Number, x::DenseMPO{L}, y::DenseMPO{L}, Alg::Algebraalgo;kwargs...) where L
    tmp = deepcopy(y)'
    
    to = TimerOutput()
    @timeit to "initialize XY Env" begin
        Envx = Environment([deepcopy(x),tmp])
        Envy = Environment([deepcopy(y),tmp])
        initialize!(Envx)
        initialize!(Envy)
    end

    info = Algebrainfo()
    while info.n ≤ Alg.N
        localto = TimerOutput()

        l2rinfo = Algebrasweepinfo(L2R())
        mto = axpby!(α,β,Envx,Envy,Alg,l2rinfo)
        merge!(localto,mto)
        merge!(info,l2rinfo)

        r2linfo = Algebrasweepinfo(R2L())
        mto = axpby!(α,β,Envx,Envy,Alg,r2linfo)
        merge!(localto,mto)
        merge!(info,r2linfo)

        show(localto;title = "axpy!")
        print("\n")
        show(info)
        merge!(to,localto)

        info.err < Alg.tol && break
    end

    @assert Envx.layer[2] == Envy.layer[2]
    return xp!(Envx.layer[2]',y),to,info
end

function axpby!(α::Number, β::Number, Envx::Environment{2}, Envy::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R};kwargs...)
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in 1:L-1
        localinfo = Algebrasiteinfo()
        x₀ = composite(Envx.layer[1].ts[site:site+1]...)
        @timeit localto "composite" ts = map(z -> contract(z.envs[site], z.layer[1].ts[site:site+1]..., z.envs[site+2]),[Envx,Envy])
        @timeit localto "SVD" tl, tc, tr, ~ = tsvd(axpby!(α, β, ts...); direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tr = contract(tc,tr) 
        # @timeit localto "SVD" tl, tr, temptruncerr = tsvd(axpby!(α,β,ts...); direction=:right,trunc = Alg.trunc)
        @timeit localto "push right" map(z -> pushright!(z, tl, tr),[Envx,Envy])
        x = composite(Envx.layer[1].ts[site:site+1]...)
        localinfo.err = norm(x-x₀)

        merge!(sweepinfo,localinfo)
    end
    return localto
end

function axpby!(α::Number, β::Number, Envx::Environment{2}, Envy::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L};kwargs...)
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in L:-1:2
        localinfo = Algebrasiteinfo()
        x₀ = composite(Envx.layer[1].ts[site-1:site]...)
        @timeit localto "composite" ts = map(z -> contract(z.envs[site-1], z.layer[1].ts[site-1:site]..., z.envs[site+1]),[Envx,Envy])
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(axpby!(α, β, ts...); direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tl = contract(tl,tc) 
        # @timeit localto "SVD" tl, tr, temptruncerr = tsvd(axpby!(α,β,ts...); direction=:left,trunc = Alg.trunc)
        @timeit localto "push left" map(z -> pushleft!(z, tl, tr),[Envx,Envy])
        x = composite(Envx.layer[1].ts[site-1:site]...)
        localinfo.err = norm(x-x₀)

        merge!(sweepinfo,localinfo)
    end
    return localto
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

function axpy!(α::Number, x::DenseMPO, y::DenseMPO;kwargs...)
    return axpby!(α,1,x,y;kwargs...)
end

function axpy!(α::Number, x::CompositeMPOTensor, y::CompositeMPOTensor)
    return axpby!(α,1,x,y)
end

function xpy!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO}
    return axpby!(1,1,x,y)
end

function xp!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO}
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



