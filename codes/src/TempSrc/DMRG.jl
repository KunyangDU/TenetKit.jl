
function DMRG2!(Env::Environment{3}, 
                D_MPS::Int64, 
                LanczosInfo::Number,
                 ;
                 Nsweep::Int64=5, 
                 trunc_tol::Float64 = 1e-5, 
                 return_error = false
    )

    ψ = Env.layer[1]
    H = Env.layer[2]
    L = Env.L

    lsE = []

    ϵ = 0
    ϵ1 = 0
    totalK = 0
    for i in 1:Nsweep

        @time "sweep $i finished, max truncation error = $(ϵ), K = $(totalK)" begin
            Eg = 0
            println(">>>>>> Right >>>>>>")
            for site in 1:L-1
                Eg,Ev,K = groundEig(projright2(Env,site),LanczosInfo)
                tl, tr, ϵ1 = tsvd(Ev; direction=:right,trunc = truncdim(D_MPS))
                pushright!(Env,tl, tr)
                ϵ = max(ϵ,ϵ1)
                totalK = max(totalK,K)
            end
            println("<<<<<< Left <<<<<<")
            for site in L:-1:2
                Eg,Ev,K = groundEig(projleft2(Env,site),LanczosInfo)
                tl, tr, ϵ1 = tsvd(Ev; direction=:left,trunc = truncdim(D_MPS))
                pushleft!(Env,tl, tr)
                ϵ = max(ϵ,ϵ1)
                totalK = max(totalK,K)
            end
            push!(lsE, Eg)
        end
        
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

function DMRG2!(ψ::DenseMPS, H::SparseMPO, D_MPS::Int64,LanczosInfo::Number = 1e-8;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    return DMRG2!(Env, D_MPS,LanczosInfo;kwargs...)
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

function DMRG1!(ψ::DenseMPS, H::SparseMPO, D_MPS::Int64,LanczosInfo::Number = 1e-8;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    return DMRG1!(Env, D_MPS,LanczosInfo;kwargs...)
end

function DMRG1!(Env::Environment{3}, 
                D_MPS::Int64, 
                LanczosInfo::Number,
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

    ϵ = 0
    ϵ1 = 0
    totalK = 0
    for i in 1:Nsweep

        @time "sweep $i finished, max truncation error = $(ϵ), K = $(totalK)" begin
            Eg = 0
            println(">>>>>> Right >>>>>>")
            for site in 1:L-1
                if cbe 
                    ϵ1 = CBE!(Env,site+1,D_MPS;kwargs...)
                    ϵ = max(ϵ,ϵ1)
                end
                Eg,Ev,K = groundEig(proj1(Env,site),LanczosInfo)
                tl,tr = leftorth(Ev)
                tr = contract(tr,Env.layer[1].ts[site+1])
                pushright!(Env,tl, tr)
                ϵ = max(ϵ,ϵ1)
                totalK = max(totalK,K)
            end
            println("<<<<<< Left <<<<<<")
            for site in L:-1:2
                if cbe 
                    ϵ1 = CBE!(Env,site-1,D_MPS;kwargs...)
                    ϵ = max(ϵ,ϵ1)
                end
                Eg,Ev,K = groundEig(proj1(Env,site),LanczosInfo)
                tl,tr = rightorth(Ev)
                tl = contract(Env.layer[1].ts[site-1],tl)
                pushleft!(Env,tl, tr)
                ϵ = max(ϵ,ϵ1)
                totalK = max(totalK,K)
            end
            push!(lsE, Eg)
        end

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


