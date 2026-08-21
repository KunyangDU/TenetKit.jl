
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
    verbose = get(kwargs,:verbose,false)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    algo = Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,N,tol,verbose,isdisk)
    return axpby!(α,x,β,y,algo;kwargs...)
end

function axpby!(α::Number, x::DenseMPO{L}, β::Number, y::DenseMPO{L}, Alg::Algebraalgo;kwargs...) where L
    y′ = y'

    to = TimerOutput()
    __init_io__()
    @timeit to "initialize XY Env" begin
        Envx = Environment([x,y′];isdisk=Alg.isdisk)
        Envy = Environment([y,y′];isdisk=Alg.isdisk)
        initialize!(Envx)
        initialize!(Envy)
    end

    info = Algebrainfo()
    try
        while info.n ≤ Alg.N
            localto = TimerOutput()

            l2rinfo = Algebrasweepinfo(L2R())
            mto = axpby!(α,Envx,β,Envy,Alg,l2rinfo)

            show(mto;title = ">>> axpby! >>>")
            print("\n")
            show(l2rinfo)
            flush(stdout)

            merge!(localto,mto)
            merge!(info,l2rinfo)

            r2linfo = Algebrasweepinfo(R2L())
            mto = axpby!(α,Envx,β,Envy,Alg,r2linfo)

            show(mto;title = "<<< axpby! <<<")
            print("\n")
            show(r2linfo)
            flush(stdout)

            merge!(localto,mto)
            merge!(info,r2linfo)

            _merge_io!(localto)
            merge!(to,localto)

            info.err < Alg.tol && break
        end
        @assert Envx.layer[2] == Envy.layer[2]
        return xp!(y′, y), to, info
    finally
        Alg.isdisk && (cleanup!(Envx); cleanup!(Envy); cleanup!(y′))
    end


end

function axpby!(α::Number, Envx::Environment{2}, β::Number, Envy::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R};kwargs...)
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in 1:L-1
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(Envx.layer[2][site:site+1]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "action" ts = map(z -> actionb(proj2(z.envs[site], nothing, nothing, z.envs[site+2]), composite(z.layer[1][site:site+1]...)), [Envx, Envy])
        @timeit localto "SVD" tl, tc, tr, localinfo.err, localinfo.bond = tsvd(axpby!(α, ts[1], β, ts[2]); direction=:center,trunc = Alg.trunc)
        @timeit localto "splice" tr = splice(tc,tr)
        @timeit localto "push right" map([Envx,Envy]) do Env
            N = length(Env.layer)
            Env.layer[N][site:site+1] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site + 1),1:N)
            pushright!(Env)
        end
        x = composite(Envx.layer[2][site:site+1]...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
    end
    return localto
end

function axpby!(α::Number, Envx::Environment{2}, β::Number, Envy::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L};kwargs...)
    localto = TimerOutput()
    L = length(Envx.layer[1])
    for site in L:-1:2
        Alg.verbose && (time₀ = time())
        localinfo = Algebrasiteinfo()
        x₀ = deepcopy(composite(Envx.layer[2][site-1:site]...))
        @assert (x2 = norm(x₀)^2) ≠ 0
        @timeit localto "action" ts = map(z -> actionb(proj2(z.envs[site-1], nothing, nothing, z.envs[site+1]), composite(z.layer[1][site-1:site]...)), [Envx, Envy])
        @timeit localto "SVD" tl, tc, tr, localinfo.err, localinfo.bond = tsvd(axpby!(α, ts[1], β, ts[2]); direction=:center,trunc = Alg.trunc)
        @timeit localto "splice" tl = splice(tl,tc)
        @timeit localto "push left" map([Envx,Envy]) do Env
            N = length(Env.layer)
            Env.layer[N][site-1:site] = adjoint.([tl, tr])
            map(v -> canonicalize!(Env.layer[v],site - 1),1:N)
            pushleft!(Env)
        end
        x = composite(Envx.layer[2][site-1:site]...)
        localinfo.err = norm(x-x₀)^2/x2

        merge!(sweepinfo,localinfo)
        Alg.verbose && vbshow(site, time₀, localinfo, Alg)
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

axpy!(α::Number, x::DenseMPO, y::DenseMPO;kwargs...) = axpby!(α,x,1,y;kwargs...)
axpy!(α::Number, x::DenseMPO, y::DenseMPO, algo::Algebraalgo;kwargs...) = axpby!(α,x,1,y,algo;kwargs...)
axpy!(α::Number, x::CompositeMPOTensor, y::CompositeMPOTensor) = axpby!(α,x,1,y)
xpy!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO} = axpby!(1,x,1,y)

function xp!(x::T, y::T) where T <: Union{DenseMPO,AdjointMPO,DenseMPS,AdjointMPS}
    y[:] = x[:]
    y.center = x.center
    return y
end

function xp!(x::T₁,y::T₂) where T₁ <: Union{AdjointMPS{L}, AdjointMPO{L}} where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L
    for i in 1:L
        y[i] = adjoint(x[i])
    end
    y.center = x.center
    return y
end

