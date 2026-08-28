
function _initialMPS(O::SparseProjectiveHamiltonian{1})
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.A[1].A, O.H[1].m[1,1].A])...)
    dom = collect(codomain(O.EnvR.A[1].A))[1]
    tmp = MPSTensor(randn,codom,dom)
    normalize!(tmp)
    return tmp
end

function _initialMPS(O::SparseProjectiveHamiltonian{2})
    codom = ⊗(map(x -> collect(domain(x))[end],[O.EnvL.A[1].A, [O.H[i].m[1,1].A for i in 1:2]...])...)
    dom = collect(codomain(O.EnvR.A[1].A))[1]
    tmp = CompositeMPSTensor(randn,codom,dom)
    normalize!(tmp)
    return tmp
end

function groundEig(O::Union{SparseProjectiveHamiltonian{N},DenseProjectiveHamiltonian{3,N}},alg::Krylovalgo = DMRGDefaultLanczos;x₀ = _initialMPS(O)) where N
    reset_timer!(get_timer("action"))
    Eg,Ev,info = eigsolve(x -> action(O,x), x₀, 1, :SR, alg.Alg)
    @assert imag(Eg[1]) < 1e-8 "Operator not hermitian"
    return real(Eg[1]), normalize(Ev[1]), Lanczosinfo(info)
end

function evolve!(
    obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
    O::Union{SparseProjectiveHamiltonian{N},DenseProjectiveHamiltonian{3,N}}, τ::Number,
    alg::Krylovalgo = TDVPDefaultLanczos) where N
    nm = normalize!(obj)
    reset_timer!(get_timer("action"))
    tmp,info = exponentiate(x -> action(O,x), -τ, obj, alg.Alg)
    rmul!(tmp,nm)
    obj.A = tmp.A
    info.normres > 1e-8 && (@warn "evolve normres > 1e-8")
    return obj, Lanczosinfo(info)
end
