
function DMRG2!(Env::Environment{3}, 
                trunc::TruncationScheme
                 ;
                 Nsweep::Int64=5, 
                 trunc_tol::Float64 = 1e-4, 
    )

    L = Env.L

    lsE = []
    
    for i in 1:Nsweep
        ϵ = 0
        ϵ1 = 0
        totalK = 0
        vns = zeros(L-1)
        lsEg = zeros(L-1)

        to = TimerOutput()
        for site in 1:L-1
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj2(Env,site,site+1)
                @timeit to "lanczos" lsEg[site],Ev,K = groundEig(projH)
            end                 
            @timeit to "SVD" tl, tr, ϵ1, vns[site] = tsvd(Ev; direction=:right,trunc = trunc)
            @timeit to "pushright" pushright!(Env,tl, tr)
            ϵ = max(ϵ,ϵ1)
            totalK = max(totalK,K)
        end
        show(to;title=">>> DMRG >>>")
        filter!(!isnan,vns)
        println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns)), Eg = $(lsEg[end]), σ(Eg) = $(std(lsEg))")
        
        to = TimerOutput()
        for site in L:-1:2
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj2(Env,site-1,site)
                @timeit to "lanczos" lsEg[site-1],Ev,K = groundEig(projH)
            end 
            @timeit to "SVD" tl, tr, ϵ1, vns[site-1] = tsvd(Ev; direction=:left,trunc = trunc)
            @timeit to "pushleft" pushleft!(Env,tl, tr)
            ϵ = max(ϵ,ϵ1)
            totalK = max(totalK,K)
        end
        show(to;title="<<< DMRG <<<")
        filter!(!isnan,vns)
        println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns)), Eg = $(lsEg[end]), σ(Eg) = $(std(lsEg))")
        push!(lsE, lsEg[1])
        
        GC.gc()

        if ϵ > trunc_tol
            return lsE
        end 
    end

    return lsE
end

function pushright!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site:site+1] = [tl,tr]
    Env.layer[3].ts[site:site+1] = adjoint(Env.layer[1].ts[site:site+1])
    pushright!(Env)
end

function pushleft!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site-1:site] = [tl, tr]
    Env.layer[3].ts[site-1:site] = adjoint(Env.layer[1].ts[site-1:site])
    pushleft!(Env)
end

function DMRG2!(ψ::DenseMPS, H::SparseMPO, trunc::TruncationScheme;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    return DMRG2!(Env, trunc;kwargs...)
end

# function DMRG2!(Env::Environment{3}, lsD::Vector{Int64};kwargs...)
#     lsinfo = Vector(undef,length(lsD))
#     for (i,D) in enumerate(lsD)
#         lsinfo[i] = @benchmark DMRG2!($Env, $D; $kwargs...)
#     end
#     return lsinfo
# end

# function DMRG2!(ψ::DenseMPS, H::SparseMPO, lsD::Vector{Int64};
#     kwargs...)
#     @time "Initialize Environment" begin
#         Env = Environment([ψ,H,adjoint(ψ)])
#         initialize!(Env)
#     end
#     lsinfo = DMRG2!(Env, lsD;kwargs...)
#     return lsinfo
# end

function DMRG1!(ψ::DenseMPS, H::SparseMPO, trunc::TensorKit.TruncationScheme;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    return DMRG1!(Env, trunc;kwargs...)
end

function DMRG1!(Env::Environment{3}, 
                trunc::TensorKit.TruncationScheme
                ;
                Nsweep::Int64=5, 
                trunc_tol::Float64 = 1e-5, 
                cbe::Bool = true,kwargs...
    )

    L = Env.L

    lsE = []

    for i in 1:Nsweep
        ϵ = 0
        totalK = 0
        vns = zeros(L-1)
        lsEg = zeros(L-1)
        to = TimerOutput()
        for site in 1:L-1
            if cbe 
                @timeit to "CBE" ϵ1 = CBE!(Env,site+1,trunc;kwargs...)
                ϵ = max(ϵ,ϵ1)
            end
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj1(Env,site)
                @timeit to "lanczos" lsEg[site],Ev,K = groundEig(projH)
            end
            @timeit to "orthogonalize" begin
                tl,tr = leftorth(Ev)
                vns[site] = vonNeumann(tr.A)
                tr = contract(tr,Env.layer[1].ts[site+1])
            end
            @timeit to "pushright" pushright!(Env,tl, tr)
            totalK = max(totalK,K)
        end
        show(to;title=">>> DMRG >>>")
        println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(filter(!isnan,vns))), σ(S) = $(std(filter(!isnan,vns))), Eg = $(lsEg[end]), σ(Eg) = $(std(lsEg))")
        to = TimerOutput()
        for site in L:-1:2
            if cbe 
                @timeit to "CBE" ϵ1 = CBE!(Env,site-1,trunc;kwargs...)
                ϵ = max(ϵ,ϵ1)
            end
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj1(Env,site)
                @timeit to "lanczos" lsEg[site-1],Ev,K = groundEig(projH)
            end
            @timeit to "orthogonalize" begin
                tl,tr = rightorth(Ev)
                vns[site-1] = vonNeumann(tl.A)
                tl = contract(Env.layer[1].ts[site-1],tl)
            end
            @timeit to "pushleft" pushleft!(Env,tl, tr)
            totalK = max(totalK,K)
        end
        show(to;title="<<< DMRG <<<")
        println("\nD = $(_maxdim(Env.layer[1])), TruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(filter(!isnan,vns))), σ(S) = $(std(filter(!isnan,vns))), Eg = $(lsEg[1]), σ(Eg) = $(std(lsEg))")
        push!(lsE, lsEg[1])

        GC.gc()

        if ϵ > trunc_tol
            return lsE
        end 
    end

    return lsE

end


#= =============================== =#


function DMRG!(Env::Environment{3,L}, Alg::DMRGalgo;kwargs...) where L
    info =  DMRGinfo()
    lsinfo = []
    lsE = []
    for _ in 1:Alg.N

        l2rinfo = DMRGsweepinfo(L2R())
        to = DMRG!(Env,Alg,l2rinfo)
        show(to;title=">>> DMRG >>>")
        print("\n")
        show(l2rinfo)
        print("\n")
        merge!(info,l2rinfo)

        r2linfo = DMRGsweepinfo(R2L())
        to = DMRG!(Env,Alg,r2linfo)
        show(to;title="<<< DMRG <<<")
        print("\n")
        show(r2linfo)
        print("\n")
        merge!(info,r2linfo)

        GC.gc()

        push!(lsinfo,deepcopy(info))
        push!(lsE,info.E)

        info.ϵ > Alg.tol && return lsinfo
    end
    return lsE,lsinfo
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{SingleSite,alg,D,tol},info::DMRGsweepinfo{L2R}) where {L,alg,D,tol}
    localto = TimerOutput()
    lsE = []
    for site in 1:L-1
        localinfo = DMRGsiteinfo()
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(L2R())
            @timeit localto "CBE" CBE!(Env, Alg, Alg.alg, cbeinfo)
            merge!(localinfo,cbeinfo)
        end
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj1(Env,site)
            @timeit localto "lanczos" localinfo.E, Egv, localinfo.solver = groundEig(projH)
        end
        @timeit localto "orthogonalize" begin
            tl,tr = leftorth(Egv)
            localinfo.bond = BondInfo(tr)
            tr = contract(tr,Env.layer[1].ts[site+1])
        end
        @timeit localto "pushright" pushright!(Env,tl, tr)
        merge!(info,localinfo)
        push!(lsE,localinfo.E)
    end

    info.σE = std(filter(!isnan,lsE))

    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{SingleSite,alg,D,tol},info::DMRGsweepinfo{R2L}) where {L,alg,D,tol}
    localto = TimerOutput()
    lsE = []
    for site in L:-1:2
        localinfo = DMRGsiteinfo()
        if alg <: CBEalgo 
            cbeinfo = CBEinfo(R2L())
            @timeit localto "CBE" CBE!(Env, Alg, Alg.alg, cbeinfo)
            merge!(localinfo,cbeinfo)
        end
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj1(Env,site)
            @timeit localto "lanczos" localinfo.E, Egv, localinfo.solver = groundEig(projH)
        end
        @timeit localto "orthogonalize" begin
            tl,tr = rightorth(Egv)
            localinfo.bond = BondInfo(tl)
            tl = contract(Env.layer[1].ts[site-1],tl)
        end
        @timeit localto "pushleft" pushleft!(Env,tl, tr)
        merge!(info,localinfo)
        push!(lsE,localinfo.E)
    end

    info.σE = std(filter(!isnan,lsE))

    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{DoubleSite,alg,D,tol},info::DMRGsweepinfo{L2R}) where {L,alg,D,tol}
    localto = TimerOutput()
    lsE = []
    for site in 1:L-1
        localinfo = DMRGsiteinfo()
        @timeit localto "Krylov" begin
            @timeit localto "projection" projH = proj2(Env,site,site+1)
            @timeit localto "lanczos" localinfo.E,Egv,K = groundEig(projH)
        end
        @timeit localto "SVD" tl, tc, tr, localinfo.ϵ = tsvd(Egv; direction=:center,trunc = truncdim(Alg.D) & truncbelow(Alg.ϵ))
        localinfo.bond = BondInfo(tc)
        @timeit localto "contract" tr = contract(tc,tr) 
        @timeit localto "pushright" pushright!(Env,tl, tr)
        merge!(info,localinfo)
        push!(lsE,localinfo.E)
    end
    info.σE = std(filter(!isnan,lsE))

    return localto
end

function DMRG!(Env::Environment{3,L},Alg::DMRGalgo{DoubleSite,alg,D,tol},info::DMRGsweepinfo{R2L}) where {L,alg,D,tol}
    localto = TimerOutput()
    lsE = []
    for site in L:-1:2
        localinfo = DMRGsiteinfo()
            @timeit localto "Krylov" begin
                @timeit localto "projection" projH = proj2(Env,site-1,site)
                @timeit localto "lanczos" localinfo.E,Egv,K = groundEig(projH)
            end 
            @timeit localto "SVD" tl, tc, tr, ϵ = tsvd(Egv; direction=:center,trunc = truncdim(Alg.D) & truncbelow(Alg.ϵ))
            localinfo.bond = BondInfo(tc)
            @timeit localto "contract" tl = contract(tl,tc) 
            @timeit localto "pushleft" pushleft!(Env,tl, tr)
        merge!(info,localinfo)
        push!(lsE,localinfo.E)
    end
    info.σE = std(filter(!isnan,lsE))

    return localto
end

