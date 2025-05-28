
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
    ts = nothing

    for i in 1:N, j in 1:M
        isnothing(O.H.ts[1].m[i,j]) && continue
        tmp = contract(O.EnvL.A[i], obj, O.H.ts[1].m[i,j])
        t = contract(tmp,O.EnvR.A[j])
        if isnothing(ts)
            ts = contract(tmp,O.EnvR.A[j])
        else
            ts += t
        end
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

function action(O::DenseProjectiveHamiltonian{2,1}, obj::DenseMPOTensor{4})
    @tensor tmp[-1 -2;-3 -4] ≔ O.EnvL.A.A[-2,1] * obj.A[-1,1,2,-4] * O.EnvR.A.A[2,-3]
    return DenseMPOTensor(tmp)
end

function action(O::SparseProjectiveHamiltonian{2}, tl::T, tr::T) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    N,M1 = O.H.D[1]
    M2,R = O.H.D[2]
    @assert M1 == M2

    ts = nothing
    for i in 1:N, j in 1:M1, k in 1:R
        isnothing(O.H.ts[1].m[i,j]) | isnothing(O.H.ts[2].m[j,k]) && continue
        EL = contract(O.EnvL.A[i], tl, O.H.ts[1].m[i,j])
        ER = contract(tr, O.H.ts[2].m[j,k],O.EnvR.A[k])
        tmp = contract(EL, ER)
        if isnothing(ts)
            ts = tmp
        else
            ts += tmp
        end
    end

    return ts
end


