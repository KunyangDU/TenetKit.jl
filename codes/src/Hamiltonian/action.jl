
function action(O::SparseProjectiveHamiltonian{0}, obj::T) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    N = O.EnvL.D
    M = O.EnvR.D
    ts = zerovector(obj,Float64)

    for i in 1:M
        tmp = contract(O.EnvL.A[i], obj)
        ts += T(contract(tmp,O.EnvR.A[i]))
    end

    return ts
end

function action(O::SparseProjectiveHamiltonian{1}, obj::Union{MPSTensor{3},DenseMPOTensor{4}})
    N,M = O.H.D[1]
    ts = zerovector(obj,Float64)

    for i in 1:N, j in 1:M
        isnothing(O.H.ts[1].m[i,j]) && continue
        tmp = contract(O.EnvL.A[i], obj, O.H.ts[1].m[i,j])
        ts += contract(tmp,O.EnvR.A[j])
    end

    return ts
end

function action(O::SparseProjectiveHamiltonian{2}, obj::Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}};svd::Bool = false)
    N,M1 = O.H.D[1]
    M2,R = O.H.D[2]
    @assert M1 == M2

    ts = zerovector(obj,Float64)
    #@show _getD(obj), N*M1*R
    for i in 1:N, j in 1:M1, k in 1:R
        isnothing(O.H.ts[1].m[i,j]) | isnothing(O.H.ts[2].m[j,k]) && continue
        if svd
            tl,tr,ϵ = tsvd(obj;direction = :left,trunc = truncdim(round(Int64,2^mean(map(x -> log(2,_getD(x)),[O.EnvL,O.EnvR])))))
            ts += contract(O.EnvL.A[i], tl, tr, O.H.ts[1].m[i,j], O.H.ts[2].m[j,k], O.EnvR.A[k])
        else
            ts += contract(contract(contract(O.EnvL.A[i], obj, O.H.ts[1].m[i,j]), O.H.ts[2].m[j,k]), O.EnvR.A[k])
        end
    end

    return ts
end



