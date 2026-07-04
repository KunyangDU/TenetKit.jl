function lanczos(action::Function, v₀::T, algo::LanczosAlgorithm) where T
    normalize!(v₀)

    info = LanczosInformation(v₀)

    for _ in 1:algo.maxdim
        @timeit info.to "lanczos!" lanczos!(action, info, algo)
        info.b[end] < algo.tol && break
    end

    length(info.a) < length(info.basis) && push!(info.a, real(inner(info.basis[end], action(info.basis[end]))))
    show(to;title = "Lanczos"); print("\n"); flush(stdout)
    return info
end

function lanczos!(action::Function, info::LanczosInformation, algo::LanczosAlgorithm)
    to = TimerOutput()
    @timeit to "action!" w,to′ = action(info.basis[end])
    merge!(to,to′;tree_point = ["action!"])
    push!(info.a, real(inner(info.basis[end], w)))
    @timeit to "orthogonalize" for _ in 1:algo.North
        @timeit to "orthogonalize" for k in eachindex(info.basis)
            schmidtorth!(w,info.basis[k])
        end
        @timeit to "fixgauge!" fixgauge!(w)
    end
    push!(info.b, norm(w)) 
    @timeit to "normalize!" normalize!(w)
    push!(info.basis, w)
    @timeit to "GC" GC.gc()
    merge!(info.to,to)
    algo.verbose && mod(length(info),algo.showtimes) == 0 && (show(to;title = "Lanczos - $(length(info)) / $(algo.maxdim)"); print("\n"); flush(stdout))
end

normalize!(A::Vector{Float64}) = (A[:] = A/norm(A))
fixgauge!(env::TaSKEnvironment) = orthogonalize!(env)
fixgauge!(::Vector{Float64}) = nothing

function schmidtorth!(w::TaSKEnvironment{L}, v::TaSKEnvironment{L}) where L
    c = inner(v, w)
    for i in 1:L
        w.TC[i] = w.TC[i] - c * v.TC[i]
    end
    return w
end