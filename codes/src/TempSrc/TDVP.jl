

function TDVP2!(Env::Environment{3}, lst::AbstractVector, D_MPS::Int64;
    LanczosInfo::Number=1e-6, TruncErr::Number=1e-4)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])

    ϵ = 0
    totalK = 0
    
    for i in 2:length(lst)
        τ = (lst[i]-lst[i-1])/2

        println("β = $(abs(lst[i]))")

        @time "sweep $i finished, max truncation error = $(ϵ), K = $(totalK)" begin
            ϵ,totalK = TDVP2!(Env, τ, D_MPS, ϵ, LanczosInfo)
        end

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
    println(">>>>>> Right >>>>>>")
    for site in 1:L-1
        tmp,K1 = evolve!(composite(Env.layer[1].ts[site:site+1]...), projright2(Env,site), τ, LanczosInfo)
        tl, tr, ϵ1 = tsvd(tmp; direction=:right,trunc = truncdim(D))
        K2 = pushright!(Env, tl, tr, τ, LanczosInfo)
        ϵ += ϵ1
        totalK = max(totalK,K1,K2)
    end
    ~,K = evolve!(Env.layer[1].ts[L], proj1(Env,L), τ, LanczosInfo)
    totalK = max(totalK,K)
    println("<<<<<< Left <<<<<<")
    for site in L:-1:2
        tmp, K1 = evolve!(composite(Env.layer[1].ts[site-1:site]...), projleft2(Env,site), τ, LanczosInfo)
        tl, tr, ϵ1 = tsvd(tmp; direction=:left,trunc = truncdim(D))
        K2 = pushleft!(Env, tl, tr, τ, LanczosInfo)
        ϵ += ϵ1
        totalK = max(totalK,K1,K2)
    end
    ~,K = evolve!(Env.layer[1].ts[1], proj1(Env,1), τ, LanczosInfo)
    totalK = max(totalK,K)
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

        @time "sweep $i finished, max truncation error = $(ϵ), K = $(totalK)" begin
            ϵ,totalK = TDVP1!(Env, τ, D_MPS, ϵ, LanczosInfo;kwargs...)
        end

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
    println(">>>>>> Right >>>>>>")
    for site in 1:L-1
        if cbe 
            B = deepcopy(Env.layer[1].ts[site+1])
            ϵ1 = CBE!(Env,site+1,D)
            splice!(Env.layer[1],B,site+1)
            Env.layer[3].ts[site] = Env.layer[1].ts[site]'
            ϵ += ϵ1
        end
        tmp,K1 = evolve!(Env.layer[1].ts[site], proj1(Env,site), τ, LanczosInfo)
        tl,tr = leftorth(tmp)
        tr = contract(tr,Env.layer[1].ts[site+1])
        K2 = pushright!(Env, tl, tr, τ, LanczosInfo)
        totalK = max(totalK,K1,K2)
    end
    ~,K = evolve!(Env.layer[1].ts[L], proj1(Env,L), τ, LanczosInfo)
    totalK = max(totalK,K)
    println("<<<<<< Left <<<<<<")
    for site in L:-1:2
        if cbe 
            A = deepcopy(Env.layer[1].ts[site-1])
            ϵ1 = CBE!(Env,site-1,D)
            splice!(Env.layer[1],A,site-1)
            Env.layer[3].ts[site] = Env.layer[1].ts[site]'
            ϵ += ϵ1
        end
        tmp, K1 = evolve!(Env.layer[1].ts[site], proj1(Env,site), τ, LanczosInfo)
        tl,tr = rightorth(tmp)
        tl = contract(Env.layer[1].ts[site-1],tl)
        K2 = pushleft!(Env, tl, tr, τ, LanczosInfo)
        ϵ += ϵ1
        totalK = max(totalK,K1,K2)
    end
    ~,K = evolve!(Env.layer[1].ts[1], proj1(Env,1), τ, LanczosInfo)
    totalK = max(totalK,K)
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


function evolve!(
    obj::Union{AbstractMPSTensor, AbstractMPOTensor, DenseMPO},
    O::SparseProjectiveHamiltonian{N}, τ::Number, LanczosInfo::Number) where N
    tmp = normalize!(obj)
    T, Q, K = MPLanczos(O,obj,LanczosInfo)
    obj.A = sum(tmp * exp(-1im*τ*T)[:,1] .* map(x->x.A, Q))
    return obj, K
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
