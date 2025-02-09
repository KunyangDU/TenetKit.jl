function SETTN!(β::Number, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L
    
    max_order = get(kwargs,:max_order,10)
    D = get(kwargs,:D,maximum(vcat(collect.(H.D)...)))
    F_tol = get(kwargs,:F_tol,1e-12)
    F = zeros(max_order)
    dF = zeros(max_order)

    Hn = deepcopy(ρ)
    dF[1] = 2*F_tol # make sure dF > F_tol
    for i in 1:max_order 
        mul!(Hn,deepcopy(Hn),H,1.,0.; D = D)
        axpy!((-β)^i / factorial(i),Hn ,ρ ; D = D)

        F[i] = - log(tr(ρ)) / 2 / β

        if i ≠ 1
            dF[i] = abs((F[i] - F[i-1]) / F[i])
            if dF[i] < F_tol && dF[i-1] < F_tol
                println("SETTN converged at $(i)th order with dF = $(dF[i-1:i])")
                break
            end
        end

        i == max_order && println("SETTN not converged at max $(i)th order with dF = $(dF)") 
    end

    return ρ
end

function SETTN!(lsβ::Vector, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L
    
    max_order = get(kwargs,:max_order,6)
    D = get(kwargs,:D,maximum(vcat(collect.(H.D)...)))
    F_tol = get(kwargs,:F_tol,1e-8)
    F = zeros(max_order)
    lsρ = Vector(undef,length(lsβ))
    lsρ = repeat([deepcopy(ρ),],length(lsβ))

    β = lsβ[end]

    Hn = deepcopy(ρ)
    dF = 2*F_tol # make sure dF > F_tol
    for i in 1:max_order 
        mul!(Hn,deepcopy(Hn),H,1.,0.; D = D)
        axpy!((-β)^i / factorial(i),Hn ,ρ ; D = D)

        F[i] = - log(tr(ρ)) /2/β
        i ≠ 1 && (dF = abs((F[i] - F[i-1]) / F[i]))

        for (iβ,βi) in enumerate(lsβ[1:end-1])
            lsρ[iβ] = axpy!((-βi)^i / factorial(i),Hn ,lsρ[iβ] ; D = D)
        end
        

        if dF < F_tol
           println("SETTN converged at $(i)th order with dF = $(dF)")
           break
        end

        i == max_order && println("SETTN not converged at max $(i)th order with dF = $(dF)") 
    end
    lsρ[end] = deepcopy(ρ)

    return lsρ
end

