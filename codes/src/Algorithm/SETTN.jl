

function SETTN1!(β::Number, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L

    trunc = get(kwargs,:trunc,notrunc())
    N = get(kwargs,:max_order,10)
    tol = get(kwargs,:tol,1e-12)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    # algo = get(kwargs,:algo,CBEalgo(dynamicSVD(1.2,2),NoStruc(),0,_getdim(trunc),isnothing(_getcutoff(trunc)) ? tol : _getcutoff(trunc)))
    algo = get(kwargs,:algo,CBEalgo(dynamicSVD(ceil(Int64,_getdim(trunc) * 1.25)),DSA(),3,_getdim(trunc)))
    multol = get(kwargs,:tol,1e-12)
    verbose = get(kwargs,:verbose,false)
    Alg = SETTNalgo(SingleSite(),Algebraalgo(SingleSite(),algo,trunc,3,multol,verbose,isdisk),trunc,N,tol)
    return SETTN!(β, H, ρ, Alg)
end

function SETTN2!(β::Number, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L

    trunc = get(kwargs,:trunc,notrunc())
    N = get(kwargs,:max_order,10)
    tol = get(kwargs,:tol,1e-12)
    isdisk = get(kwargs,:isdisk,IS_DISK[])
    verbose = get(kwargs,:verbose,false)
    multol = get(kwargs,:tol,1e-12)
    alg = SETTNalgo(DoubleSite(),Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,3,multol,verbose,isdisk),trunc,N,tol)
    return SETTN!(β, H, ρ, alg)
end

function SETTN!(β::Number,H::SparseMPO{L}, ρ::DenseMPO, Alg::SETTNalgo) where L

    to = TimerOutput()
    info = SETTNinfo()
    __init_io__()

    Hn = deepcopy(ρ)
    try
        @timeit to "I - βH" begin
            localto,localinfo = SETTN!(β,H,Hn,ρ,1,Alg)
        end
        merge!(to,localto, tree_point = ["I - βH"])
        merge!(info,localinfo)

        _merge_io!(localto)
        show(localto;title = "SETTN - (I - βH)")
        print("\n")
        show(localinfo)

        flush(stdout)

        while info.n < Alg.N
            @timeit to "Iteration" localto,localinfo = SETTN!(β,H,Hn,ρ,info.n + 1,Alg)
            info.err = abs((localinfo.lnZ - info.lnZ) / localinfo.lnZ)
            info.lnZ = localinfo.lnZ
            merge!(info,localinfo)
            merge!(to,localto, tree_point = ["Iteration"])

            _merge_io!(localto)
            show(localto;title = "SETTN - $(info.n) (≤$(Alg.N))")
            println("\n")
            show(localinfo)
            println("dF = $(info.err)")
            flush(stdout)

            if info.err < Alg.tol
                println("SETTN converged at $(info.n)-th order with dF = $(info.err)")
                break
            end

            if info.n == Alg.N
                println("SETTN not converged at max $(info.n)th order with dF = $(info.err)")
                break
            end

            GC.gc()
        end
        flush(stdout)

        _merge_io!(to)
        show(to;title = "SETTN")
        print("\n")

        return ρ
    finally
        cleanup!(Hn)
    end
end

function SETTN!(β::Number,H::SparseMPO{L},Hn::DenseMPO,ρ::DenseMPO,order::Int64,Alg::SETTNalgo) where L
    to = TimerOutput()
    info = SETTNsweepinfo()
    
    _merge_io!(to)
    @timeit to "mul!" ~,multo,minfo = mul!(Hn,Hn,H,1,Alg.alg)
    _merge_io!(to)
    # @timeit to "axpy!" ~,axpyto,ainfo = axpy!((-β) ^ order / factorial(order),Hn ,ρ , Alg.alg)
    @timeit to "axpy!" ~,axpyto,ainfo = axpy!((-β) ^ order / factorial(order),Hn ,ρ ; trunc = Alg.trunc, tol = Alg.tol, verbose = Alg.alg.verbose, isdisk = Alg.alg.isdisk)
    @timeit to "calculate lnZ" info.lnZ = log(tr(ρ))
    # F = - log(tr(ρ)) / 2 / β

    merge!(to,multo, tree_point = ["mul!"])
    merge!(to,axpyto, tree_point = ["axpy!"])
    merge!(info.bond,minfo.bond)
    merge!(info.bond,ainfo.bond)
    info.err = minfo.err
    return to,info
end

# @timeit to "mul!" ~,multo,minfo = mul!(Hn,deepcopy(Hn),H; trunc = Alg.trunc, tol = Alg.tol)
# @timeit to "mul!" ~,multo,minfo = mul!(Hn,deepcopy(Hn),H,1,0,Alg.alg)
# @timeit to "axpy!" ~,axpyto,ainfo = axpby!((-β) ^ order / factorial(order),Hn ,1,ρ,Alg.alg)

# function SETTN!(β::Number,H::SparseMPO{L},Hn::DenseMPO,ρ::DenseMPO,order::Int64,Alg::SETTNalgo{DoubleSite}) where L
#     to = TimerOutput()
#     info = SETTNsweepinfo()
#     @timeit to "mul!" ~,multo,minfo = mul!(Hn,deepcopy(Hn),H; trunc = Alg.trunc, tol = Alg.tol)
#     @timeit to "axpy!" ~,axpyto,ainfo = axpy!((-β) ^ order / factorial(order),Hn ,ρ ; trunc = Alg.trunc, tol = Alg.tol)
#     merge!(to,multo, tree_point = ["mul!"])
#     merge!(to,axpyto, tree_point = ["axpy!"])
#     @timeit to "calculate lnZ" info.lnZ = log(tr(ρ))
#     # F = - log(tr(ρ)) / 2 / β
#     merge!(info.bond,minfo.bond)
#     merge!(info.bond,ainfo.bond)
#     info.err = minfo.err
#     return to,info
# end



