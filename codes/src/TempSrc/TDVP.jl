

function TDVP2!(Env::Environment{3}, lst::AbstractVector, D_MPS::Int64;
    LanczosInfo::Number=1e-6, TruncErr::Number=1e-4)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])

    ϵ = 0
    totalK = 0
    
    for i in 2:length(lst)
        τ = (lst[i]-lst[i-1])/2

        println("t = $(abs(lst[i]))")

        ϵ,totalK = TDVP2!(Env, τ, D_MPS, ϵ, LanczosInfo)

        ϵ > TruncErr && break
        push!(lsobj,deepcopy(Env.layer[1]))
        
    end

    return lsobj
end

function TDVP2!(ψ::DenseMPS, H::SparseMPO, t::Number, Nt::Int64, D_MPS::Int64;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    lst = collect(range(0,t,Nt))
    lsψ = TDVP2!(Env, lst, D_MPS;kwargs...)
    return lsψ, lst
end

function TDVP2!(Env::Environment{3}, τ::Number, D::Int64, ϵ::Number, LanczosInfo::Number)
    L = Env.L
    ϵ1 = 0
    totalK = 0
    vns = zeros(L-1)
    to = TimerOutput()
    for site in 1:L-1
        @timeit to "evolve" tmp,K1 = evolve!(composite(Env.layer[1].ts[site:site+1]...), projright2(Env,site), τ, LanczosInfo)
        @timeit to "SVD" tl, tr, ϵ1, vns[site] = tsvd(tmp; direction=:right,trunc = truncdim(D))
        @timeit to "pushright" K2 = pushright!(Env, tl, tr, τ, LanczosInfo)
        ϵ += ϵ1
        totalK = max(totalK,K1,K2)
    end
    @timeit to "backevolve" ~,K = evolve!(Env.layer[1].ts[L], proj1(Env,L), τ, LanczosInfo)
    totalK = max(totalK,K)
    show(to;title=">>> TDVP >>>")
    filter!(!isnan,vns)
    println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
    to = TimerOutput()
    for site in L:-1:2
        @timeit to "evolve" tmp, K1 = evolve!(composite(Env.layer[1].ts[site-1:site]...), projleft2(Env,site), τ, LanczosInfo)
        @timeit to "SVD" tl, tr, ϵ1, vns[site-1] = tsvd(tmp; direction=:left,trunc = truncdim(D))
        @timeit to "pushright" K2 = pushleft!(Env, tl, tr, τ, LanczosInfo)
        ϵ += ϵ1
        totalK = max(totalK,K1,K2)
    end
    @timeit to "backevolve" ~,K = evolve!(Env.layer[1].ts[1], proj1(Env,1), τ, LanczosInfo)
    totalK = max(totalK,K)
    show(to;title="<<< TDVP <<<")
    filter!(!isnan,vns)
    println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")

    GC.gc()
    return ϵ, totalK
end

function TDVP1!(Env::Environment{3}, lst::AbstractVector, D_MPS::Int64;
    LanczosInfo::Number=1e-6, TruncErr::Number=1e-4,kwargs...)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])

    ϵ = 0
    totalK = 0
    
    for i in 2:length(lst)
        τ = (lst[i]-lst[i-1])/2

        println("β = $(abs(lst[i]))")

        ϵ,totalK = TDVP1!(Env, τ, D_MPS, ϵ, LanczosInfo;kwargs...)

        ϵ > TruncErr && break
        push!(lsobj,deepcopy(Env.layer[1]))

    end

    return lsobj
end

function TDVP1!(ψ::DenseMPS, H::SparseMPO, t::Number, Nt::Int64, D_MPS::Int64;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,ψ'])
        initialize!(Env)
    end
    lst = collect(range(0,t,Nt))
    lsψ = TDVP1!(Env, lst, D_MPS;kwargs...)
    return lsψ, lst
end

function TDVP1!(Env::Environment{3}, τ::Number, D::Int64, ϵ::Number, LanczosInfo::Number;cbe::Bool=true)
    L = Env.L
    ϵ1 = 0
    totalK = 0
    vns = zeros(L-1)
    to = TimerOutput()
    for site in 1:L-1
        if cbe 
            @timeit to "CBE" begin
                B = deepcopy(Env.layer[1].ts[site+1])
                ϵ1 = CBE!(Env,site+1,D)
                splice!(Env.layer[1],B,site+1)
                Env.layer[3].ts[site] = Env.layer[1].ts[site]'
                ϵ += ϵ1
            end
        end
        @timeit to "evolve" tmp,K1 = evolve!(Env.layer[1].ts[site], proj1(Env,site), τ, LanczosInfo)
        @timeit to "orthogonalize" begin
            tl,tr = leftorth(tmp)
            vns[site] = vonNeumann(tr.A)
            tr = contract(tr,Env.layer[1].ts[site+1])
        end 
        @timeit to "pushright" K2 = pushright!(Env, tl, tr, τ, LanczosInfo)
        totalK = max(totalK,K1,K2)
    end
    @timeit to "backevolve" ~,K = evolve!(Env.layer[1].ts[L], proj1(Env,L), τ, LanczosInfo)
    totalK = max(totalK,K)
    show(to;title=">>> TDVP >>>")
    filter!(!isnan,vns)
    println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
    for site in L:-1:2
        if cbe 
            @timeit to "CBE" begin
                A = deepcopy(Env.layer[1].ts[site-1])
                ϵ1 = CBE!(Env,site-1,D)
                splice!(Env.layer[1],A,site-1)
                Env.layer[3].ts[site] = Env.layer[1].ts[site]'
                ϵ += ϵ1
            end
        end
        @timeit to "evolve" tmp, K1 = evolve!(Env.layer[1].ts[site], proj1(Env,site), τ, LanczosInfo)
        @timeit to "orthogonalize" begin
            tl,tr = rightorth(tmp)
            vns[site-1] = vonNeumann(tl.A)
            tl = contract(Env.layer[1].ts[site-1],tl)
        end
        @timeit to "pushleft" K2 = pushleft!(Env, tl, tr, τ, LanczosInfo)
        ϵ += ϵ1
        totalK = max(totalK,K1,K2)
    end
    @timeit to "backevolve" ~,K = evolve!(Env.layer[1].ts[1], proj1(Env,1), τ, LanczosInfo)
    totalK = max(totalK,K)
    show(to;title="<<< TDVP <<<")
    filter!(!isnan,vns)
    println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
    GC.gc()
    return ϵ, totalK
end

function pushright!(Env::Environment{3}, tl::Union{AbstractMPSTensor, AbstractMPOTensor}, tr::Union{AbstractMPSTensor, AbstractMPOTensor}, τ::Number, LanczosInfo::Number)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site] = tl
    Env.layer[3].ts[site] = adjoint(Env.layer[1].ts[site])
    pushright!(Env)
    ~, K = evolve!(tr, proj1(Env,site+1), -τ, LanczosInfo)
    Env.layer[1].ts[site+1] = tr
    Env.layer[3].ts[site+1] = adjoint(Env.layer[1].ts[site+1])

    map(x -> Env.layer[x].center .+= 1,[1,3])
    return K
end

function pushleft!(Env::Environment{3}, tl::Union{AbstractMPSTensor, AbstractMPOTensor}, tr::Union{AbstractMPSTensor, AbstractMPOTensor}, τ::Number, LanczosInfo::Number)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site] = tr
    Env.layer[3].ts[site] = adjoint(Env.layer[1].ts[site])
    pushleft!(Env)
    ~, K = evolve!(tl, proj1(Env,site-1), -τ, LanczosInfo)
    Env.layer[1].ts[site-1] = tl
    Env.layer[3].ts[site-1] = adjoint(Env.layer[1].ts[site-1])

    map(x -> Env.layer[x].center .-= 1,[1,3])
    return K
end


#= function evolve!(
    obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
    O::SparseProjectiveHamiltonian{N}, τ::Number, LanczosInfo::Number) where N
    tmp = normalize!(obj)
    T, Q, K = MPLanczos(O,obj,LanczosInfo)
    obj.A = sum(tmp * exp(-1im*τ*T)[:,1] .* map(x->x.A, Q))
    return obj, K
end =#

function evolve!(
    obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
    O::SparseProjectiveHamiltonian{N}, τ::Number, LanczosInfo::Number) where N
    tmp = normalize!(obj)
    obj,info = exponentiate(x -> action(O,x),-1im * τ,obj,TDVPDefaultLanczos)
    rmul!(obj,tmp)
    @assert info.residual ≈ 0
    return obj, info.numiter
end

function tanTRG2!(ρ::DenseMPO, H::SparseMPO, lsβ::AbstractVector, D::Int64, LanczosInfo::Number=1e-5;kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ρ,H,ρ'])
        initialize!(Env)
    end
    lsρ = TDVP2!(Env,lsβ .* (-1im), D;LanczosInfo=LanczosInfo,kwargs...)
    return lsρ
end

function tanTRG1!(ρ::DenseMPO, H::SparseMPO, lsβ::AbstractVector, D::Int64, LanczosInfo::Number=1e-5;kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ρ,H,ρ'])
        initialize!(Env)
    end
    lsρ = TDVP1!(Env,lsβ .* (-1im), D;LanczosInfo=LanczosInfo,kwargs...)
    return lsρ
end
