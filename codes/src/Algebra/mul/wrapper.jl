function mul!(C::Union{DenseMPO,DenseMPS}, A::Union{DenseMPO,DenseMPS}, B::Union{DenseMPO,SparseMPO}, α::Number, trunc::TruncationScheme;kwargs...)
    D = _getdim(trunc)
    ϵ = _getcutoff(trunc)

    Nmul = get(kwargs,:Nmul,3)
    verbose = get(kwargs,:verbose,false)
    alg = get(kwargs,:alg,Algebraalgo(SingleSite(),CBEalgo(dynamicSVD(ceil(Int64, D * 1.25)),NoStruc(),0,D),trunc,Nmul,ϵ,verbose))
    
    return mul!(C,A,B,α,alg)
end