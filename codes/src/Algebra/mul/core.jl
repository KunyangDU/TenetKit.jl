function mul!(C::Union{DenseMPO{L₁},DenseMPS{L₁}}, A::T₁, B::T₂, α::Number, Alg::Algebraalgo; kwargs...) where T₁ <: Union{DenseMPO{L₁},DenseMPS{L₁}} where T₂ <: Union{DenseMPO{L₂},SparseMPO{L₂},AdjointMPO{L₂},RefMPO{L₂}} where {L₁,L₂}

    @assert L₁ == L₂
    to = TimerOutput()
    __init_io__()
    A′ = ref(T₁)(A,adjoint)
    @timeit to "initialize ABC Env" begin
        EnvAB = Environment([C,B,A′];isdisk=Alg.isdisk)
        initialize!(EnvAB;kwargs...)
    end

    info = Algebrainfo()
    try
        while info.n ≤ Alg.N
            info.err = 0
            localto = TimerOutput()

            l2rinfo = Algebrasweepinfo(L2R())
            mto = mul!(EnvAB,α,Alg,l2rinfo)

            show(mto;title = ">>> mul! - $(info.n) / $(Alg.N) >>>")
            print("\n")
            show(l2rinfo)
            flush(stdout)

            merge!(localto,mto)
            merge!(info,l2rinfo)

            r2linfo = Algebrasweepinfo(R2L())
            mto = mul!(EnvAB,α,Alg,r2linfo)

            show(mto;title = "<<< mul! - $(info.n) / $(Alg.N) <<<")
            print("\n")
            show(r2linfo)
            flush(stdout)

            merge!(localto,mto)
            merge!(info,r2linfo)

            _merge_io!(localto)
            merge!(to,localto)

            max(l2rinfo.err,r2linfo.err) < Alg.tol && break
        end
        return C, to, info
    finally
        Alg.isdisk && cleanup!(EnvAB)
    end

end