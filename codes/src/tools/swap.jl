function leftswap!(obj::DenseMPS{L}, site::Int64) where L 
    sl = site - 1
    sr = site

    Al₀,Ar₀ = obj[sl].A, obj[sr].A
    D = dim(space(Al₀)[3])
    @tensor A[-1,-2,-3;-4] ≔ Al₀[-1,-2,1] * Ar₀[1,-3,-4] 
    Al,C,Ar = svd(A,(1,3),(2,4);trunc = truncdim(D))
    obj[sl] = MPSTensor(Al*C)
    obj[sr] = MPSTensor(permute(Ar, ((1, 2), (3,))))

    obj.center .-= 1
    return obj
end

function rightswap!(obj::DenseMPS{L}, site::Int64) where L 
    sl = site
    sr = site + 1

    Al₀,Ar₀ = obj[sl].A, obj[sr].A
    D = dim(space(Al₀)[3])
    @tensor A[-1,-2,-3;-4] ≔ Al₀[-1,-2,1] * Ar₀[1,-3,-4] 
    Al,C,Ar = svd(A,(1,3),(2,4);trunc = truncdim(D))
    obj[sl] = MPSTensor(Al)
    obj[sr] = MPSTensor(permute(C*Ar, ((1, 2), (3,))))

    obj.center .+= 1
    return obj
end

function swap!(obj::DenseMPS{L}, site::Int64) where L
    center = obj.center[1]
    site == center && return obj

    if site < center
        while center > site
            leftswap!(obj,center)
            center -= 1
        end
    end

    if site > center
        while center < site
            rightswap!(obj,center)
            center += 1
        end
    end

    return obj
end