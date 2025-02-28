function bDMRG2!(ψ::DenseMPS, H::SparseMPO, D_MPS::Int64,LanczosInfo::Number;
    kwargs...)
    @time "Initialize Environment" begin
        Env = Environment([ψ,H,adjoint(ψ)])
        initialize!(Env)
    end
    ~,truncerror = DMRG2!(Env, D_MPS, LanczosInfo;kwargs...,trunc_tol = 1e-16,return_error = true)
    println("Benchmark begin with truncerror = $(truncerror)")
    lsE,btrial = bDMRG2!(Env, D_MPS, LanczosInfo;kwargs...)
    return lsE,btrial
end

function bDMRG2!(Env::Environment{3}, D_MPS::Int64, LanczosInfo::Number=1e-8)

    ψ = Env.layer[1]
    H = Env.layer[2]
    L = Env.L

    lsE = []

    btrial = @benchmark begin
        Eg = 0
        for site in 1:$L-1
            Eg,Ev = groundEig(projright2($Env,site),$LanczosInfo)
            tl, tr, ~ = tsvd(Ev; direction=:right,trunc = truncdim($D_MPS))
            pushright!($Env,tl, tr)
        end
        for site in $L:-1:2
            Eg,Ev = groundEig(projleft2($Env,site),$LanczosInfo)
            tl, tr, ~ = tsvd(Ev; direction=:left,trunc = truncdim($D_MPS))
            pushleft!($Env,tl, tr)
        end
        push!($lsE, Eg)
    end
    
    GC.gc()

    return lsE,btrial
end



