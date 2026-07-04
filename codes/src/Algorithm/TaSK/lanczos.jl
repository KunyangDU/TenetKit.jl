function lanczos(action::Function, v₀; maxdim::Int=100, tol::Float64=1e-12)
    normalize!(v₀)

    basis = [v₀]
    a = Float64[]
    b = Float64[]

    for i in 1:maxdim
        w = action(basis[i])

        push!(a, real(inner(basis[i], w)))

        for _ in 1:2
            for k in 1:i
                w = w - inner(basis[k], w) * basis[k]
            end
            fixgauge!(w)
        end
        push!(b, norm(w)) 
        normalize!(w)

        @show i,b[end]

        b[end] < tol && break
        push!(basis, w)
    end

    if length(a) < length(basis)
        w_last = action(basis[end])
        push!(a, real(inner(basis[end], w_last)))
    end

    return (a=a, b=b, basis=basis, converged = b[end] < tol)
end

function laneig(result, n::Int=length(result.a))
    T = diagm(0 => result.a, 1 => result.b, -1 => conj(result.b))
    F = eigen(T)
    return F.values[1:n], F.vectors[:, 1:n]
end


normalize!(A::Vector{Float64}) = (A[:] = A/norm(A))
fixgauge!(env::TaSKEnvironment) = orthogonalize!(env)
fixgauge!(::Vector{Float64}) = nothing