using LinearAlgebra
using SparseArrays
using SpecialFunctions: besselj

# ==========================================
# 1. 构造 Toy Model 和初始态
# ==========================================
N = 1000 # 矩阵维度
println("=> 正在初始化 $N x $N 随机 Hermitian 矩阵...")
# 构造一个随机的稀疏 Hermitian 矩阵作为 H
H_raw = sprandn(N, N, 0.05) 
H = Symmetric(H_raw + H_raw')

# 随机初态，并归一化
v = randn(ComplexF64, N)
normalize!(v)

# 设置物理时间步长
τ = 0.5 

# ==========================================
# 2. 模拟 AI 预测：获取精确能谱边界
# ==========================================
# 对于普通小矩阵，我们直接求特征值来获取边界
vals = eigvals(Matrix(H))
λ_min, λ_max = vals[1], vals[end]
W = λ_max - λ_min
println("=> 能谱边界: [$(round(λ_min, digits=2)), $(round(λ_max, digits=2))]")
println("=> 谱宽 W = $(round(W, digits=2)), 无量纲参数 τW = $(round(τ*W, digits=2))")

# ==========================================
# 3. 精确对角化 (ED) 计算基准答案
# ==========================================
println("=> 正在计算精确演化 (exp(-iτH) * v)...")
v_exact = exp(-1im * τ * Matrix(H)) * v

# ==========================================
# 4. Chebyshev 展开法实现
# ==========================================
function chebyshev_evolve(H, v, τ, λ_min, λ_max; tol=1e-10)
    W = λ_max - λ_min
    λ_bar = (λ_max + λ_min) / 2
    τ_tilde = τ * W / 2

    # 预计算 Bessel 系数
    coeffs = ComplexF64[]
    push!(coeffs, besselj(0, τ_tilde))
    k = 1
    while true
        ak = 2 * (-1im)^k * besselj(k, τ_tilde)
        push!(coeffs, ak)
        # 截断条件：系数绝对值极小，且 k 已经超过了振荡区 τ_tilde
        if abs(ak) < tol && k > abs(τ_tilde)
            break
        end
        k += 1
    end
    m = length(coeffs)
    println("=> Chebyshev 截断阶数 m = $m")

    # 定义 Action 函数
    H_action(u) = H * u
    # 映射到 [-1, 1] 的 Action: H_tilde * v = (2Hv - λ_bar*v) / W
    H_tilde_action(u) = (2 .* H_action(u) .- 2λ_bar .* u) ./ W

    # 三项递推开始 (只占用 3 个向量的内存)
    v0 = copy(v)
    v1 = H_tilde_action(v)
    
    result = coeffs[1] .* v0 .+ coeffs[2] .* v1
    
    for i in 2:(m-1)
        v_new = 2 .* H_tilde_action(v1) .- v0
        result .+= coeffs[i+1] .* v_new
        
        # 滚动更新历史向量
        v0 = v1
        v1 = v_new
    end
    
    # 乘上整体相移
    return exp(-1im * τ * λ_bar) .* result
end

println("=> 正在计算 Chebyshev 演化...")
v_cheb = chebyshev_evolve(H, v, τ, λ_min, λ_max, tol=1e-10)

# ==========================================
# 5. 精度验证
# ==========================================
# 保真度 F = |<v_exact | v_cheb>|
fidelity = abs(dot(v_exact, v_cheb))
# L2 误差
err_norm = norm(v_exact - v_cheb)

println("-"^40)
println("验证结果：")
println("保真度 Fidelity : ", fidelity)
println("L2 误差 Norm    : ", err_norm)
if err_norm < 1e-8
    println("=> 测试通过！Chebyshev 方法完美复现了矩阵指数演化。")
else
    println("=> 误差偏大，请检查容差 tol 或能谱边界。")
end

# 运行结果：
# => 正在初始化 1000 x 1000 随机 Hermitian 矩阵...
# => 能谱边界: [-20.12, 20.03]
# => 谱宽 W = 40.15, 无量纲参数 τW = 20.07
# => 正在计算精确演化 (exp(-iτH) * v)...
# => 正在计算 Chebyshev 演化...
# => Chebyshev 截断阶数 m = 30
# ----------------------------------------
# 验证结果：
# 保真度 Fidelity : 1.0000000000000284
# L2 误差 Norm    : 2.5062618636244056e-12
# => 测试通过！Chebyshev 方法完美复现了矩阵指数演化。