
function DMRG2!(Env::Environment{3}, 
                D_MPS::Int64
                 ;
                 Nsweep::Int64=5, 
                 trunc_tol::Float64 = 1e-4, 
                 return_error = false
    )

    ψ = Env.layer[1]
    H = Env.layer[2]
    L = Env.L

    lsE = []
    
    for i in 1:Nsweep
        ϵ = 0
        ϵ1 = 0
        totalK = 0
        vns = zeros(L-1)

        Eg = 0
        to = TimerOutput()
        for site in 1:L-1
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj2(Env,site,site+1)
                @timeit to "lanczos" Eg,Ev,K = groundEig(projH)
            end                 
            @timeit to "SVD" tl, tr, ϵ1, vns[site] = tsvd(Ev; direction=:right,trunc = truncdim(D_MPS))
            @timeit to "pushright" pushright!(Env,tl, tr)
            ϵ = max(ϵ,ϵ1)
            totalK = max(totalK,K)
        end
        show(to;title=">>> DMRG >>>")
        filter!(!isnan,vns)
        println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
        
        to = TimerOutput()
        for site in L:-1:2
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj2(Env,site-1,site)
                @timeit to "lanczos" Eg,Ev,K = groundEig(projH)
            end 
            @timeit to "SVD" tl, tr, ϵ1, vns[site-1] = tsvd(Ev; direction=:left,trunc = truncdim(D_MPS))
            @timeit to "pushleft" pushleft!(Env,tl, tr)
            ϵ = max(ϵ,ϵ1)
            totalK = max(totalK,K)
        end
        show(to;title="<<< DMRG <<<")
        filter!(!isnan,vns)
        println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns)), Eg = $(Eg)")
        push!(lsE, Eg)
        
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

function DMRG2!(ψ::DenseMPS, H::SparseMPO, D_MPS::Int64;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    return DMRG2!(Env, D_MPS;kwargs...)
end

function DMRG2!(Env::Environment{3}, lsD::Vector{Int64};kwargs...)
    lsinfo = Vector(undef,length(lsD))
    for (i,D) in enumerate(lsD)
        lsinfo[i] = @benchmark DMRG2!($Env, $D; $kwargs...)
    end
    return lsinfo
end

function DMRG2!(ψ::DenseMPS, H::SparseMPO, lsD::Vector{Int64};
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    lsinfo = DMRG2!(Env, lsD;kwargs...)
    return lsinfo
end

function DMRG1!(ψ::DenseMPS, H::SparseMPO, D_MPS::Int64;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    return DMRG1!(Env, D_MPS;kwargs...)
end

function DMRG1!(Env::Environment{3}, 
                D_MPS::Int64
                ;
                Nsweep::Int64=5, 
                trunc_tol::Float64 = 1e-5, 
                return_error = false,
                cbe::Bool = true,kwargs...
    )

    ψ = Env.layer[1]
    H = Env.layer[2]
    L = Env.L

    lsE = []

    
    for i in 1:Nsweep
        ϵ = 0
        ϵ1 = 0
        totalK = 0
        vns = zeros(L-1)

        Eg = 0
        to = TimerOutput()
        for site in 1:L-1
            if cbe 
                @timeit to "CBE" ϵ1 = CBE!(Env,site+1,D_MPS;kwargs...)
                ϵ = max(ϵ,ϵ1)
            end
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj1(Env,site)
                @timeit to "lanczos" Eg,Ev,K = groundEig(projH)
            end
            @timeit to "orthogonalize" begin
                tl,tr = leftorth(Ev)
                vns[site] = vonNeumann(tr.A)
                tr = contract(tr,Env.layer[1].ts[site+1])
            end
            @timeit to "pushright" pushright!(Env,tl, tr)
            ϵ = max(ϵ,ϵ1)
            totalK = max(totalK,K)
        end
        show(to;title=">>> DMRG >>>")
        filter!(!isnan,vns)
        println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
        to = TimerOutput()
        for site in L:-1:2
            if cbe 
                @timeit to "CBE" ϵ1 = CBE!(Env,site-1,D_MPS;kwargs...)
                ϵ = max(ϵ,ϵ1)
            end
            @timeit to "Krylov" begin
                @timeit to "projection" projH = proj1(Env,site)
                @timeit to "lanczos" Eg,Ev,K = groundEig(projH)
            end
            @timeit to "orthogonalize" begin
                tl,tr = rightorth(Ev)
                vns[site-1] = vonNeumann(tl.A)
                tl = contract(Env.layer[1].ts[site-1],tl)
            end
            @timeit to "pushleft" pushleft!(Env,tl, tr)
            ϵ = max(ϵ,ϵ1)
            totalK = max(totalK,K)
        end
        show(to;title="<<< DMRG <<<")
        filter!(!isnan,vns)
        println("\nTruncErr = $(ϵ), K = $(totalK), ⟨S⟩ = $(mean(vns)), σ(S) = $(std(vns))")
        push!(lsE, Eg)

        GC.gc()

        if ϵ > trunc_tol
            if return_error
                return lsE,ϵ
            else
                return lsE
            end
        end 
    end

    if return_error
        return lsE,ϵ
    else
        return lsE
    end

end


