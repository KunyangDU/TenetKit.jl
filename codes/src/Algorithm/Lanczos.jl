
function groundEig(Oprs::Vector{AbstractTensorMap{ComplexSpace,2,2}},
    EnvL::AbstractTensorMap{ComplexSpace,2,1},EnvR::AbstractTensorMap{ComplexSpace,1,2},
    LanczosLevel::Int64)

    q1 = let 
    Tensor(rand, ComplexF64, ⊗(codomain(EnvL)[1],let 
        spaces = []
        for h in Oprs 
            push!(spaces,collect(codomain(h))[2])
        end
        spaces
        end...,codomain(EnvR)[1])) |> x -> x' / norm(x)
    end

    T, Q = trivialLanczos(Oprs,EnvL,EnvR,q1,LanczosLevel)
    λ, v = eigen(T)
    Eg,Ev = argmin(real.(λ)) |> x -> (real.(λ)[x], sum(Q .* v[:, x]))
    return Eg, Ev / norm(Ev)
end

########### normalized ##########


function trivialLanczos(Oprs::Vector,
    EnvL::AbstractTensorMap,EnvR::AbstractTensorMap,
    q1::AbstractTensorMap,
    LanczosLevel::Int64;kwargs...)
    Q = Vector{AbstractTensorMap}(undef, LanczosLevel)
    α = zeros(LanczosLevel)
    β = zeros(LanczosLevel-1)

    Q[1] = q1


    for j = 1:LanczosLevel
        if j == 1
            w = EnvMerge(EnvL,Q[j],Oprs...,EnvR)
        else
            w = EnvMerge(EnvL,Q[j],Oprs...,EnvR) - β[j-1] * Q[j-1]
        end

        α[j] = ApproxReal((w*Q[j]')[1])
        w -= α[j] * Q[j]
        
        if j < LanczosLevel
            β[j] = norm(w)
            if β[j] ≈ 0
                @error "flow interrupted"
            else
                Q[j+1] = w / β[j]
            end
        end
    end
    
    T = diagm(0 => α) +diagm(-1 => β) + diagm(1 => β)
    return T, Q
    
end

function groundEig(Oprs::Vector{Union{AbstractTensorMap{ComplexSpace,1,3},AbstractTensorMap{ComplexSpace,2,2}}},
    EnvL::AbstractTensorMap{ComplexSpace,2,1},EnvR::AbstractTensorMap{ComplexSpace,2,1},
    LanczosLevel::Int64)

    q1 = let 
        Tensor(rand, ComplexF64, ⊗(codomain(EnvL)[1],let 
            if typeof(Oprs[1]) <: AbstractTensorMap{ComplexSpace,1,3}
                map(x -> collect(codomain(x))[1],Oprs)
            elseif typeof(Oprs[2]) <: AbstractTensorMap{ComplexSpace,1,3}
                map(x -> collect(codomain(x))[end],Oprs)
            else
                @error "Oprs form error"
            end
        end...,codomain(EnvR)[1])) |> x -> x' / norm(x)
    end

    T, Q = trivialLanczos(Oprs,EnvL,EnvR,q1,LanczosLevel)
    λ, v = eigen(T)
    Eg,Ev = argmin(real.(λ)) |> x -> (real.(λ)[x], sum(Q .* v[:, x]))
    return Eg, Ev / norm(Ev)
end

function Evolve(
    localψ::AbstractTensorMap{ComplexSpace,2,4},
    Oprs::Vector{Union{AbstractTensorMap{ComplexSpace,1,3},AbstractTensorMap{ComplexSpace,2,2}}},
    EnvL::AbstractTensorMap{ComplexSpace,2,1},EnvR::AbstractTensorMap{ComplexSpace,2,1},
    τ::Number,
    LanczosLevel::Int64)
    q1 = localψ / norm(localψ)
    T, Q = trivialLanczos(Oprs,EnvL,EnvR,q1,LanczosLevel)
    A = sum(norm(localψ) * exp(-1im*τ*T)[:,1] .* Q)
    return A
end

function Evolve(
    localψ::AbstractTensorMap{ComplexSpace,0,4},
    Oprs::Vector{Union{AbstractTensorMap{ComplexSpace,1,3},AbstractTensorMap{ComplexSpace,2,2}}},
    EnvL::AbstractTensorMap{ComplexSpace,2,1},EnvR::AbstractTensorMap{ComplexSpace,2,1},
    τ::Number,
    LanczosLevel::Int64)
    q1 = localψ / norm(localψ)
    T, Q = trivialLanczos(Oprs,EnvL,EnvR,q1,LanczosLevel)
    A = sum(norm(localψ) * exp(-1im*τ*T)[:,1] .* Q)
    return A
end

function Evolve(
    localψ::Union{AbstractTensorMap{ComplexSpace,0,3},AbstractTensorMap{ComplexSpace,1,3}},
    Opr::AbstractTensorMap{ComplexSpace,1,3},
    EnvL::AbstractTensorMap{ComplexSpace,2,1},EnvR::AbstractTensorMap{ComplexSpace,2,1},
    τ::Number,
    LanczosLevel::Int64)
    q1 = localψ / norm(localψ)
    T, Q = trivialLanczos([Opr],EnvL,EnvR,q1,LanczosLevel)
    A = sum(norm(localψ) * exp(-1im*τ*T)[:,1] .* Q)
    return A
end

