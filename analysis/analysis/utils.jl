function easyinterp10(v,N=100)
    return 10. .^ (range(log10.(extrema(v))..., N))
end
gaussian(ω::Float64, ωs::Vector, Ss::Vector;σ = 0.5) = sum(@. Ss * exp(-(ω - ωs)^2/σ^2/2))/sqrt(2pi)/σ

function cfe(a::Vector, b::Vector, d::Float64, ω_grid; K=20, η=1e-3)
    M = length(a)
    K = min(K, M-1)

    # 尾部: 对角化子矩阵
    T_tail = Symmetric(diagm(0 => a[K+1:end], 1 => b[K+1:end-1], -1 => b[K+1:end-1]))
    λ_tail, V_tail = eigen(T_tail)
    S_tail = abs2.(V_tail[1,:])

    A = zeros(length(ω_grid))
    for (idx, ω) in enumerate(ω_grid)
        # 尾部 Green 函数: R(z) = Σ S_β / (z - λ_β)
        R = sum(S_tail[β] / (ω + 1im*η - λ_tail[β]) for β in eachindex(λ_tail))
        # 向前递推
        for j in K:-1:1
            R = 1.0 / (ω + 1im*η - a[j] - b[j]^2 * R)
        end
        A[idx] = -d^2 * imag(R) / π
    end
    return A
end
