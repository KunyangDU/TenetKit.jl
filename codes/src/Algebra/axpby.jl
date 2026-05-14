
"""
axpy!(α, x, y) -> y
Overwrite y with x * α + y and return y. If x and y have the same axes, it's equivalent with y .+= x .* a.
# kwargs
D_MPO: MPO bond dimension. Default is the maximum D of x, y.
Nsweep: times of variational calculation (sweep). Default is 2.

"""
function axpby!(α::Number, x::DenseMPO{L}, β::Number, y::DenseMPO{L};kwargs...) where L
    trunc = get(kwargs,:trunc,notrunc())
    N  = get(kwargs,:N,3)
    tol = get(kwargs,:tol,1e-8)
    algo = Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,N,tol)
    return axpby!(α,x,β,y,algo;kwargs...)
end

function axpby!(α::Number, x::DenseMPO{L}, β::Number, y::DenseMPO{L}, Alg::Algebraalgo;kwargs...) where L
    y′ = y'
    
    to = TimerOutput()
    @timeit to "initialize XY Env" begin
        Envx = Environment([deepcopy(x),y′])
        Envy = Environment([y,y′])
        initialize!(Envx)
        initialize!(Envy)
    end

    info = Algebrainfo()
    while info.n ≤ Alg.N
        localto = TimerOutput()

        l2rinfo = Algebrasweepinfo(L2R())
        mto = axpby!(α,Envx,β,Envy,Alg,l2rinfo)
        merge!(localto,mto)
        merge!(info,l2rinfo)

        r2linfo = Algebrasweepinfo(R2L())
        mto = axpby!(α,Envx,β,Envy,Alg,r2linfo)
        merge!(localto,mto)
        merge!(info,r2linfo)

        show(localto;title = "axpby!")
        print("\n")
        show(info)
        merge!(to,localto)

        info.err < Alg.tol && break
    end

    @assert Envx.layer[2] == Envy.layer[2]
    return xp!(y′',y),to,info
end

function axpby!(α::Number, Envx::Environment{2}, β::Number, Envy::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R};kwargs...)
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in 1:L-1
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(Envx.layer[2][site:site+1]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "composite" ts = map(z -> contract(z.envs[site], z.layer[1][site:site+1]..., z.envs[site+2]),[Envx,Envy])
        @timeit localto "SVD" tl, tc, tr, ~ = tsvd(axpby!(α, ts[1], β, ts[2]); direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tr = contract(tc,tr) 
        @timeit localto "push right" map([Envx,Envy]) do Env
            N = length(Env.layer)
            Env.layer[N][site:site+1] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site + 1),1:N)
            pushright!(Env)
        end
        x = composite(Envx.layer[2][site:site+1]...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
    end
    return localto
end

function axpby!(α::Number, Envx::Environment{2}, β::Number, Envy::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L};kwargs...)
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in L:-1:2
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(Envx.layer[2][site-1:site]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "composite" ts = map(z -> contract(z.envs[site-1], z.layer[1][site-1:site]..., z.envs[site+1]),[Envx,Envy])
        @timeit localto "SVD" tl, tc, tr, localinfo.err = tsvd(axpby!(α, ts[1], β, ts[2]); direction=:center,trunc = Alg.trunc)
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tl = contract(tl,tc) 
        @timeit localto "push left" map([Envx,Envy]) do Env
            N = length(Env.layer)
            Env.layer[N][site-1:site] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site - 1),1:N)
            pushleft!(Env)
        end
        x = composite(Envx.layer[2][site-1:site]...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
    end
    return localto
end

function axpby!(α::Number, Envx::Environment{2}, β::Number, Envy::Environment{2}, Alg::Algebraalgo{SingleSite,alg}, sweepinfo::Algebrasweepinfo{L2R};kwargs...) where alg
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in 1:L-1
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(Envx.layer[1][site:site+1]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(L2R())
            @timeit localto "CBE_X" cbetoX = CBE!(Envx, CBEalgo(Alg.alg,DA(),2), cbeinfo)
            @timeit localto "CBE_Y" cbetoY = CBE!(Envy, CBEalgo(Alg.alg,DA(),2), cbeinfo)
            tLX₀,tRX₀ = Envx.layer[2][site:site+1]
            tLY₀,tRY₀ = Envy.layer[2][site:site+1]
            @timeit localto "after-orthogonalize" orthogonalize!(tRY₀,tRX₀,L2R())
            @timeit localto "direct-sum" tR = _cbedsum(tRY₀,tRX₀,L2R())
            @timeit localto "splice" tLY = splice(tLY₀,tRY₀,tR,L2R())
            @timeit localto "splice" tLX = splice(tLX₀,tRX₀,tR,L2R())
            Envx.layer[2][site:site+1] .= tLX,tR 
            Envy.layer[2][site:site+1] .= tLY,tR 
            map([Envx,Envy]) do env
                env.envs[site+1] = pushleft(map(x -> env.layer[x],1:length(env.layer))...,env.envs[site+2],site+1)
            end
            merge!(localinfo,cbeinfo)
            merge!(localto,cbetoX,tree_point = ["CBE_X"])
            merge!(localto,cbetoY,tree_point = ["CBE_Y"])
        end
        ts = map([Envx,Envy]) do Env 
            @timeit localto "projection" projH = proj1(Env,site)
            action(projH,Env.layer[1][site])
        end

        @timeit localto "orthogonalize" begin
            tl,tr = leftorth(axpby!(α, ts[1], β, ts[2]))
            localinfo.bond = BondInfo(tr)
            tr = contract(tr,Envx.layer[2][site+1]')
        end
        @timeit localto "push right" map([Envx,Envy]) do Env
            N = length(Env.layer)
            Env.layer[N][site:site+1] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site + 1),1:N)
            pushright!(Env)
        end
        x = composite(Envx.layer[1][site:site+1]...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
    end
    return localto
end

function axpby!(α::Number, Envx::Environment{2}, β::Number, Envy::Environment{2}, Alg::Algebraalgo{SingleSite,alg}, sweepinfo::Algebrasweepinfo{R2L};kwargs...) where alg
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in L:-1:2
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(Envx.layer[1][site-1:site]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(R2L())
            @timeit localto "CBE_X" cbetoX = CBE!(Envx, CBEalgo(Alg.alg,DA(),2), cbeinfo)
            @timeit localto "CBE_Y" cbetoY = CBE!(Envy, CBEalgo(Alg.alg,DA(),2), cbeinfo)
            tLX₀,tRX₀ = Envx.layer[2][site-1:site]
            tLY₀,tRY₀ = Envy.layer[2][site-1:site]
            @timeit localto "after-orthogonalize" orthogonalize!(tLX₀,tLY₀,R2L())
            @timeit localto "direct-sum" tL = _cbedsum(tLX₀,tLY₀,R2L())
            @timeit localto "splice" tRX = splice(tLX₀,tRX₀,tL,R2L())
            @timeit localto "splice" tRY = splice(tLY₀,tRY₀,tL,R2L())
            Envx.layer[2][site-1:site] .= tL,tRX 
            Envy.layer[2][site-1:site] .= tL,tRY 
            map([Envx,Envy]) do env
                env.envs[site] = pushright(map(x -> env.layer[x],1:length(env.layer))...,env.envs[site-1],site-1)
            end
            merge!(localinfo,cbeinfo)
            merge!(localto,cbetoX,tree_point = ["CBE_X"])
            merge!(localto,cbetoY,tree_point = ["CBE_Y"])
        end
        ts = map([Envx,Envy]) do Env 
            @timeit localto "projection" projH = proj1(Env,site)
            action(projH,Env.layer[1][site])
        end
        @timeit localto "orthogonalize" begin
            tl,tr = rightorth(axpby!(α, ts[1], β, ts[2]))
            localinfo.bond = BondInfo(tl)
            tl = contract(Envx.layer[2][site-1]',tl)
        end
        @timeit localto "push left" map([Envx,Envy]) do Env
            N = length(Env.layer)
            Env.layer[N][site-1:site] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site - 1),1:N)
            pushleft!(Env)
        end
        x = composite(Envx.layer[1][site-1:site]...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
    end
    return localto
end

function axpby!(α::Number, x::CompositeMPOTensor{N₁,R₁}, β::Number, y::CompositeMPOTensor{N₂,R₂}) where {N₁,R₁,N₂,R₂}
    @assert N₁ == N₂ && R₁ == R₂
    y.A = x.A * α + y.A * β
    return y
end

function axpby!(::Number, ::Nothing, β::Number, y::CompositeMPOTensor)
    y.A = y.A * β
    return y
end

function axpy!(α::Number, x::DenseMPO, y::DenseMPO;kwargs...)
    return axpby!(α,x,1,y;kwargs...)
end

function axpy!(α::Number, x::CompositeMPOTensor, y::CompositeMPOTensor)
    return axpby!(α,x,1,y)
end

function xpy!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO}
    return axpby!(1,x,1,y)
end

function xp!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS}
    # Per-element copy to avoid materializing all tensors at once (OOM risk).
    for i in 1:length(x.ts)
        y[i] = x[i]
    end
    y.center = x.center
    return y
end

