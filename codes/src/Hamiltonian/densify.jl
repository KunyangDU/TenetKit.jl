
function densify(O::SparseMPO{L};kwargs...) where L
    ps = pspace(O)
    O′ = IdDenseMPO(ps, repeat([trivial(ps),], L+1))
    canonicalize!(O′,1)
    trunc = get(kwargs, :trunc, truncdim(maximum(length.(O[:]))))
    # algo = Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,3,1e-12,false,false)
    algo = Algebraalgo(SingleSite(),CBEalgo(dynamicSVD(1.2,2),DSA(),3,_getdim(trunc)),trunc,3,1e-12,false,false)
    mul!(O′,O′,O,1,algo)
    return O′
end

function pspace(H::SparseMPO{L}) where L 
    for i in 1:L
        tmp = filter(x -> !isnothing(x.A), H[i].A)
        isempty(tmp) && continue 
        return space(tmp[1].A)[1]
    end
end