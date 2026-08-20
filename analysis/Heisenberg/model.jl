

function window(x::Number)
    if 1/2 < x ≤ 1
        return -2*(x-1)^3
    elseif 0 ≤ x ≤ 1/2
        return 1 - 6x^2 + 6x^3
    elseif -1/2 ≤ x < 0
        return 1 - 6x^2 - 6x^3
    elseif -1 ≤ x < -1/2
        return 2*(x+1)^3
    else
        return 0
    end
end

function getCorrMat(Latt::AbstractLattice,data::Dict,dataonsite = nothing;selected_point = 1:size(Latt))
    # L = length(selected_point)
    # A = zeros(L,L)
    # for i in 1:L,j in i+1:L
    #     A[i,j] = data[(selected_point[i],selected_point[j])]
    # end
    # A = zeros(size(Latt),size(Latt))
    # for i in selected_point, j in selected_point
    #     i >= j && continue
    #     A[i,j] = data[(i,j)]
    # end
    A = zeros(size(Latt),size(Latt))
    for i in selected_point, j in selected_point
        i >= j && continue
        A[i,j] = real(data[(i,j)])
    end
    A = A + A'
    if !isnothing(dataonsite)
        if typeof(dataonsite) <: Dict
            for i in 1:L 
                A[i,i] = dataonsite[(i,)]
            end
        elseif typeof(dataonsite) <: Vector
            A .+= diagm(dataonsite)
        elseif typeof(dataonsite) <: Number 
            # A .+= diagm(dataonsite*ones(L))
            tmp = zeros(size(Latt))
            tmp[selected_point] .= 1
            A .+= diagm(dataonsite*tmp)
        end
    end
    return A
end

function currentindex2(J::Matrix, h::Vector)
    ans = []
    
    ϵ = zeros(3,3,3)
    ϵ[1,2,3] = ϵ[2,3,1] = ϵ[3,1,2] = 1
    ϵ[3,2,1] = ϵ[1,3,2] = ϵ[2,1,3] = -1
    
    for α in 1:3,β in 1:3,γ in 1:3,γ′ in 1:3
        j′ = J[α,β] * h[γ] * ϵ[γ,α,γ′]
        j′ ≠ 0 && (push!(ans,(j′,(γ′,β)),(-j′,(β,γ′))))
    end
    return ans
end
