# function TDVP2!(ψ::DenseMPS, H::SparseMPO, t::Number, Nt::Int64, trunc::TruncationScheme;
#     kwargs...)
#     @time "Initialize Environment" begin
#         Env = Environment([ψ,H,adjoint(ψ)])
#         initialize!(Env)
#     end
#     lst = collect(range(0,t,Nt))
#     lsψ = TDVP2!(Env, lst, trunc;kwargs...)
#     return lsψ, lst
# end

# function TDVP2!(Env::Environment{3}, lst::AbstractVector, trunc::TruncationScheme;
#     TruncErr::Number=1e-4)

#     lsobj = Vector(undef,1)
#     lsobj[1] = deepcopy(Env.layer[1])

#     ϵ = 0
#     totalK = 0
    
#     for i in 2:length(lst)
#         τ = (lst[i]-lst[i-1])/2

#         println("t = $(abs(lst[i]))")

#         ϵ,totalK = TDVP2!(Env, τ, trunc, ϵ)

#         ϵ > TruncErr && break
#         push!(lsobj,deepcopy(Env.layer[1]))
#     end

#     return lsobj
# end

# function TDVP2!(Env::Environment{3}, τ::Number, trunc::TruncationScheme, ϵ::Number)
#     L = Env.L
#     ϵ1 = 0
#     totalK = 0
#     vns = zeros(L-1)
#     to = TimerOutput()
#     for site in 1:L-1
#         @timeit to "evolve" tmp,K1 = evolve!(composite(Env.layer[1].ts[site:site+1]...), proj2(Env,site,site+1), τ)
#         @timeit to "SVD" tl, tr, ϵ1, vns[site] = tsvd(tmp; direction=:right,trunc = trunc)
#         to1,K2 = pushright!(Env, tl, tr, τ)
#         ϵ += ϵ1
#         totalK = max(totalK,K1.numiter,K2.numiter)
#         merge!(to,to1)
#     end
#     @timeit to "evolve" ~,K = evolve!(Env.layer[1].ts[L], proj1(Env,L), τ)
#     Env.layer[3].ts[L] = Env.layer[1].ts[L]'
#     totalK = max(totalK,K.numiter)
#     show(to;title=">>> TDVP >>>")
#     filter!(!isnan,vns)
#     println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
#     to = TimerOutput()
#     for site in L:-1:2
#         @timeit to "evolve" tmp, K1 = evolve!(composite(Env.layer[1].ts[site-1:site]...), proj2(Env,site-1,site), τ)
#         @timeit to "SVD" tl, tr, ϵ1, vns[site-1] = tsvd(tmp; direction=:left,trunc = trunc)
#         to1,K2 = pushleft!(Env, tl, tr, τ)
#         ϵ += ϵ1
#         totalK = max(totalK,K1.numiter,K2.numiter)
#         merge!(to,to1)
#     end
#     @timeit to "evolve" ~,K = evolve!(Env.layer[1].ts[1], proj1(Env,1), τ)
#     Env.layer[3].ts[1] = Env.layer[1].ts[1]'
#     totalK = max(totalK,K.numiter)
#     show(to;title="<<< TDVP <<<")
#     filter!(!isnan,vns)
#     println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")

#     GC.gc()
#     return ϵ, totalK
# end

# # """
# # standard 2 site TDVP with a slightly higher error practically.
# # """
# # function TDVP2!_std(Env::Environment{3}, τ::Number, D::Int64, ϵ::Number)
# #     L = Env.L
# #     ϵ1 = 0
# #     totalK = 0
# #     vns = zeros(L-1)
# #     to = TimerOutput()
# #     for site in 1:L-2
# #         @timeit to "evolve" tmp,K1 = evolve!(composite(Env.layer[1].ts[site:site+1]...), proj2(Env,site,site+1), τ)
# #         @timeit to "SVD" tl, tr, ϵ1, vns[site] = tsvd(tmp; direction=:right,trunc = truncdim(D))
# #         to1,K2 = pushright!(Env, tl, tr, τ)
# #         ϵ += ϵ1
# #         totalK = max(totalK,K1,K2)
# #         merge!(to,to1)
# #     end
# #     @timeit to "evolve" tmp,K = evolve!(composite(Env.layer[1].ts[L-1:L]...), proj2(Env,L-1,L), 2τ)
# #     @timeit to "SVD" tl, tr, ϵ1, vns[L-1] = tsvd(tmp; direction=:left,trunc = truncdim(D))
# #     @timeit to "back evolve" evolve!(tl,proj1(Env,L-1),-τ)
# #     Env.layer[1].ts[L-1:L] = [tl,tr]
# #     Env.layer[3].ts[L-1:L] = [tl',tr']
# #     totalK = max(totalK,K)
# #     show(to;title=">>> TDVP >>>")
# #     filter!(!isnan,vns)
# #     println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
# #     to = TimerOutput()
# #     for site in L-1:-1:2
# #         @timeit to "evolve" tmp, K1 = evolve!(composite(Env.layer[1].ts[site-1:site]...), proj2(Env,site-1,site), τ)
# #         @timeit to "SVD" tl, tr, ϵ1, vns[site] = tsvd(tmp; direction=:left,trunc = truncdim(D))
# #         to1,K2 = pushleft!(Env, tl, tr, τ)
# #         ϵ += ϵ1
# #         totalK = max(totalK,K1,K2)
# #         merge!(to,to1)
# #     end
# #     @timeit to "evolve" ~,K = evolve!(Env.layer[1].ts[1], proj1(Env,1), τ)
# #     Env.layer[3].ts[1] = Env.layer[1].ts[1]'
# #     totalK = max(totalK,K)
# #     show(to;title="<<< TDVP <<<")
# #     filter!(!isnan,vns)
# #     println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns[2:end])), σ(S) = $(std(vns[2:end]))")

# #     GC.gc()
# #     return ϵ, totalK
# # end

# function TDVP1!(Env::Environment{3}, lst::AbstractVector, trunc::TruncationScheme;
#     TruncErr::Number=1e-4,kwargs...)

#     lsobj = Vector(undef,1)
#     lsobj[1] = deepcopy(Env.layer[1])

#     ϵ = 0
#     totalK = 0
    
#     for i in 2:length(lst)
#         τ = (lst[i]-lst[i-1])/2

#         println("t = $(abs(lst[i]))")

#         ϵ,totalK = TDVP1!(Env, τ, trunc, ϵ;kwargs...)

#         #ϵ > TruncErr && break
#         push!(lsobj,deepcopy(Env.layer[1]))

#     end

#     return lsobj
# end

# function TDVP1!(ψ::DenseMPS, H::SparseMPO, t::Number, Nt::Int64, trunc::TruncationScheme;
#     kwargs...)
#     @time "Initialize Environment" begin
#         Env = Environment([ψ,H,ψ'])
#         initialize!(Env)
#     end
#     lst = collect(range(0,t,Nt))
#     lsψ = TDVP1!(Env, lst, trunc;kwargs...)
#     return lsψ, lst
# end

# function TDVP1!(Env::Environment{3}, τ::Number, trunc::TruncationScheme, ϵ::Number;cbe::Bool=true)
#     L = Env.L
#     ϵ1 = 0
#     totalK = 0
#     vns = zeros(L-1)
#     to = TimerOutput()
#     for site in 1:L-1
#         if cbe 
#             @timeit to "CBE" begin
#                 B = deepcopy(Env.layer[1].ts[site+1])
#                 ϵ1 = CBE!(Env,site+1,trunc)
#                 splice!(Env.layer[1],B,site+1)
#                 Env.layer[3].ts[site] = Env.layer[1].ts[site]'
#                 ϵ += ϵ1
#             end
#         end
#         @timeit to "evolve" tmp,K1 = evolve!(Env.layer[1].ts[site], proj1(Env,site), τ)
#         @timeit to "orthogonalize" begin
#             tl,tr = leftorth(tmp)
#             vns[site] = vonNeumann(tr.A)
#         end 
#         to1,K2 = pushright!(Env,tl,tr,τ)
#         totalK = max(totalK,K1.numiter,K2.numiter)
#         merge!(to,to1)
#     end
#     @timeit to "evolve" ~,K = evolve!(Env.layer[1].ts[L], proj1(Env,L), τ)
#     Env.layer[3].ts[L] = Env.layer[1].ts[L]'
#     totalK = max(totalK,K.numiter)
    
#     show(to; title=">>> TDVP >>>")
#     filter!(!isnan,vns)
#     println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
#     to = TimerOutput()
    
#     for site in L:-1:2
#         if cbe 
#             @timeit to "CBE" begin
#                 A = deepcopy(Env.layer[1].ts[site-1])
#                 ϵ1 = CBE!(Env,site-1,trunc)
#                 splice!(Env.layer[1],A,site-1)
#                 Env.layer[3].ts[site] = Env.layer[1].ts[site]'
#                 ϵ += ϵ1
#             end
#         end
#         @timeit to "evolve" tmp, K1 = evolve!(Env.layer[1].ts[site], proj1(Env,site), τ)
#         @timeit to "orthogonalize" begin
#             tl,tr = rightorth(tmp)
#             vns[site-1] = vonNeumann(tl.A)
#         end
#         to1,K2 = pushleft!(Env,tl,tr,τ)
#         totalK = max(totalK,K1.numiter,K2.numiter)
#         merge!(to,to1)
#     end
#     @timeit to "evolve" ~,K = evolve!(Env.layer[1].ts[1], proj1(Env,1), τ)
#     Env.layer[3].ts[1] = Env.layer[1].ts[1]'
#     totalK = max(totalK,K.numiter)
    
#     show(to;title="<<< TDVP <<<")
#     filter!(!isnan,vns)
#     println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
#     GC.gc()
#     return ϵ, totalK
# end

# function tanTRG2!(ρ::DenseMPO, H::SparseMPO, lsβ::AbstractVector, trunc::TruncationScheme;kwargs...)
#     @time "Initialize Environment" begin
#         Env = Environment([ρ,H,ρ'])
#         initialize!(Env)
#     end
#     lsρ = TDVP2!(Env,lsβ .* (-1im), trunc;kwargs...)
#     return lsρ
# end

# function tanTRG1!(ρ::DenseMPO, H::SparseMPO, lsβ::AbstractVector, trunc::TruncationScheme;kwargs...)
#     @time "Initialize Environment" begin
#         Env = Environment([ρ,H,ρ'])
#         initialize!(Env)
#     end
#     lsρ = TDVP1!(Env,lsβ .* (-1im), trunc;kwargs...)
#     return lsρ
# end

#= ================================ =#

function TDVP1!(Env::Environment{3}, lst::AbstractVector;D::Int64)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])
    info = TDVPinfo()
    lsinfo = []
    alg = TDVPalgo(SingleSite(),CBEalgo(randSVD(),1.2),D,1e-6,0,1e-4,TDVPDefaultLanczos)
    
    for i in 2:length(lst)
        τ = (lst[i]-lst[i-1])/2
        println("t = $(abs(lst[i]))")
        alg.τ = τ
        
        TDVP!(Env, alg, info)

        info.ϵ > alg.tol && break
        push!(lsobj,deepcopy(Env.layer[1]))
        push!(lsinfo,deepcopy(info))
    end

    return lsobj,lsinfo
end

function TDVP2!(Env::Environment{3}, lst::AbstractVector;D::Int64)

    lsobj = Vector(undef,1)
    lsobj[1] = deepcopy(Env.layer[1])
    info = TDVPinfo()
    lsinfo = []
    alg = TDVPalgo(DoubleSite(),NoAlgorithm(),D,1e-6,0,1e-4,TDVPDefaultLanczos)
    
    for i in 2:length(lst)
        τ = (lst[i]-lst[i-1])/2
        println("t = $(abs(lst[i]))")
        alg.τ = τ
        
        TDVP!(Env, alg, info)

        info.ϵ > alg.tol && break
        push!(lsobj,deepcopy(Env.layer[1]))
        push!(lsinfo,deepcopy(info))
    end

    return lsobj,lsinfo
end

function TDVP!(Env::Environment{3,L}, Alg::TDVPalgo, info::TDVPinfo;kwargs...) where L

    l2rinfo = TDVPsweepinfo(L2R())
    to = TDVP!(Env,Alg,l2rinfo)
    show(to;title=">>> TDVP >>>")
    print("\n")
    show(l2rinfo)
    print("\n")
    merge!(info,l2rinfo)

    r2linfo = TDVPsweepinfo(R2L())
    to = TDVP!(Env,Alg,r2linfo)
    show(to;title="<<< TDVP <<<")
    print("\n")
    show(r2linfo)
    print("\n")
    merge!(info,r2linfo)

    GC.gc()
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{DoubleSite},info::TDVPsweepinfo{L2R}) where L
    localto = TimerOutput()
    for site in 1:L-1
        localinfo = TDVPsiteinfo()
        @timeit localto "evolve" tmp,localinfo.solver = evolve!(composite(Env.layer[1].ts[site:site+1]...), proj2(Env,site,site+1), Alg.τ, Alg.solver)
        @timeit localto "SVD" tl, tc, tr, localinfo.ϵ = tsvd(tmp; direction=:center,trunc = truncdim(Alg.D) & truncbelow(Alg.ϵ))
        localinfo.bond = BondInfo(tc)
        tr = contract(tc,tr)
        to,solver = pushright!(Env, tl, tr, Alg.τ, Alg.solver)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end    
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[L], proj1(Env,L), Alg.τ, Alg.solver)
    Env.layer[3].ts[L] = Env.layer[1].ts[L]'
    merge!(info.solver, solver)
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{DoubleSite},info::TDVPsweepinfo{R2L}) where L
    localto = TimerOutput()
    for site in L:-1:2
        localinfo = TDVPsiteinfo()
        @timeit localto "evolve" tmp, localinfo.solver = evolve!(composite(Env.layer[1].ts[site-1:site]...), proj2(Env,site-1,site), Alg.τ, Alg.solver)
        @timeit localto "SVD" tl, tc, tr, ϵ = tsvd(tmp; direction=:center,trunc = truncdim(Alg.D) & truncbelow(Alg.ϵ))
        localinfo.bond = BondInfo(tc)
        tl = contract(tl,tc)
        to,solver = pushleft!(Env, tl, tr, Alg.τ, Alg.solver)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[1], proj1(Env,1), Alg.τ)
    Env.layer[3].ts[1] = Env.layer[1].ts[1]'
    merge!(info.solver, solver)
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{SingleSite,alg},info::TDVPsweepinfo{L2R}) where {L,alg}
    localto = TimerOutput()
    for site in 1:L-1
        localinfo = TDVPsiteinfo()
        if alg <: CBEalgo 
            @timeit localto "CBE" begin
                cbeinfo = CBEinfo(L2R())
                B = deepcopy(Env.layer[1].ts[site+1])
                CBE!(Env,Alg,Alg.alg,cbeinfo)
                splice!(Env.layer[1],B,site+1)
                Env.layer[3].ts[site] = Env.layer[1].ts[site]'
                merge!(localinfo,cbeinfo)
            end
        end
        @timeit localto "evolve" tmp,localinfo.solver = evolve!(Env.layer[1].ts[site], proj1(Env,site), Alg.τ, Alg.solver)
        @timeit localto "orthogonalize" begin
            tl,tr = leftorth(tmp)
            localinfo.bond = BondInfo(tr)
        end 
        to,solver = pushright!(Env,tl,tr,Alg.τ,Alg.solver)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[L], proj1(Env,L), Alg.τ, Alg.solver)
    Env.layer[3].ts[L] = Env.layer[1].ts[L]'
    merge!(info.solver, solver)
    return localto
end

function TDVP!(Env::Environment{3,L},Alg::TDVPalgo{SingleSite,alg},info::TDVPsweepinfo{R2L}) where {L,alg}
    localto = TimerOutput()
    for site in L:-1:2
        localinfo = TDVPsiteinfo()
        if alg <: CBEalgo 
            @timeit localto "CBE" begin
                cbeinfo = CBEinfo(R2L())
                A = deepcopy(Env.layer[1].ts[site-1])
                CBE!(Env,Alg,Alg.alg,cbeinfo)
                splice!(Env.layer[1],A,site-1)
                Env.layer[3].ts[site] = Env.layer[1].ts[site]'
                merge!(localinfo,cbeinfo)
            end
        end
        @timeit localto "evolve" tmp, localinfo.solver = evolve!(Env.layer[1].ts[site], proj1(Env,site), Alg.τ, Alg.solver)
        @timeit localto "orthogonalize" begin
            tl,tr = rightorth(tmp)
            localinfo.bond = BondInfo(tl)
        end
        to,solver = pushleft!(Env,tl,tr,Alg.τ)
        merge!(localinfo.solver,solver)
        merge!(info,localinfo)
        merge!(localto,to)
    end
    @timeit localto "evolve" ~,solver = evolve!(Env.layer[1].ts[1], proj1(Env,1), Alg.τ, Alg.solver)
    Env.layer[3].ts[1] = Env.layer[1].ts[1]'
    merge!(info.solver, solver)
    return localto
end

function TDVP1!(ψ::DenseMPS,H::SparseMPO,t::Number,Nt::Number,D::Int64)
    @time "initialize environment" begin 
        Env = Environment([ψ,H,ψ'])
        initialize!(Env)
    end
    lst = range(0,t,Nt)
    lsobj,lsinfo = TDVP1!(Env,1im * lst;D=D)
    return lst,lsobj,lsinfo
end

function TDVP2!(ψ::DenseMPS,H::SparseMPO,t::Number,Nt::Number,D::Int64)
    @time "initialize environment" begin 
        Env = Environment([ψ,H,ψ'])
        initialize!(Env)
    end
    lst = range(0,t,Nt)
    lsobj,lsinfo = TDVP2!(Env,1im * lst;D=D)
    return lst,lsobj,lsinfo
end


function tanTRG1!(ρ::DenseMPO,H::SparseMPO,lsβ::Vector,D::Int64)
    @time "initialize environment" begin 
        Env = Environment([ρ,H,ρ'])
        initialize!(Env)
    end
    lsobj,lsinfo = TDVP1!(Env, - lsβ;D=D)
    return lsobj,lsinfo
end
function tanTRG2!(ρ::DenseMPO,H::SparseMPO,lsβ::Vector,D::Int64)
    @time "initialize environment" begin 
        Env = Environment([ρ,H,ρ'])
        initialize!(Env)
    end
    lsobj,lsinfo = TDVP2!(Env, - lsβ;D=D)
    return lsobj,lsinfo
end

