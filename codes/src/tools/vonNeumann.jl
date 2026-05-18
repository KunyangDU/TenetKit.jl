function vonNeumann!(obj::DenseMPS{L}, sites::Vector{Int64}) where L
    2length(sites) > L && (sites = setdiff(1:L,sites))

    sites = sort(sites)
    N = sum(2(sites .- collect(1:length(sites))) .+ vcat(0,ones(length(sites)-1)))
    N′ = let sites = reverse(sites)
        sum(2(L .- sites[2:end] .- collect(2:length(sites)) .+ 1) + ones(length(sites)-1)) + L - 1
    end

    if true
        for (i,s) in enumerate(sites)
            canonicalize!(obj,s)
            swap!(obj,i)
        end
        _,S,_ = svd(obj[obj.center[1]].A,(1,2),(3,))
    else
        sites = reverse(sites)
        for (i,s) in enumerate(sites)
            canonicalize!(obj,s)
            swap!(obj, L - i + 1)
        end
        _,S,_ = svd(obj[obj.center[1]].A,(1,),(2,3))
    end
    
    return vonNeumann(S)
end

function vonNeumann(obj::DenseMPS, sites::Vector{Int64})
    obj′ = deepcopy(obj)
    result = vonNeumann!(obj′, sites)
    cleanup!(obj′)
    return result
end


function vonNeumann(S::AbstractTensorMap{<:ElementarySpace,1,1})
    _tmptrace(x) = @tensor x[1,1]
    d = sqrt(_tmptrace(S*S'))
    @assert d != 0
    A = S/d |> x -> x*x'
    return real(_tmptrace(-A*log(A)))
end
