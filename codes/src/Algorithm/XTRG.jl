
function XTRG!(obj::DenseMPO{L}, Alg::XTRGalgo, info::XTRGinfo) where L
    sweepinfo = XTRGsweepinfo()
    to = TimerOutput()
    merge_io!(to)

    @timeit to "mul!" _,multo,mulinfo = mul!(obj,obj,RefMPO(obj,adjoint),1,Alg.alg;verbose = true)
    merge!(to,multo,tree_point=["mul!"])
    merge!(sweepinfo,mulinfo)
    @timeit to "normalize!" sweepinfo.lnZ = 2log(normalize!(obj)) + 2 * info.lnZ
    @timeit to "measure E" !isnothing(Alg.H) && (sweepinfo.E = real(_scalar(Environment([obj,Alg.H]))))

    @timeit to "GC" GC.gc()
    merge_io!(to)
    show(to;title = "XTRG - $(info.n) / $(Alg.N)")
    print("\n")
    show(sweepinfo)
    flush(stdout)
    merge!(info,sweepinfo)
    return obj,to,sweepinfo
end

function XTRG!(obj::DenseMPO{L}, Alg::XTRGalgo, lnZ::Float64 = 2log(normalize!(obj))) where L
    info = XTRGinfo(lnZ)
    lsinfo = XTRGsweepinfo[]
    while info.n ≤ Alg.N
        _,_,sweepinfo = XTRG!(obj,Alg,info)
        push!(lsinfo,sweepinfo)
        info.n += 1
    end
    return obj,lsinfo
end

function XTRG1!(obj::DenseMPO{L},H::SparseMPO{L},N::Int64;kwargs...) where L
    truncscheme = get(kwargs,:trunc,notrunc())
    tol = get(kwargs,:tol,1e-12)
    Nsweep = get(kwargs,:Nsweep,20)
    cbealgo = CBEalgo(dynamicSVD(1.2,2),DDA(),3,_getdim(truncscheme))
    algo = Algebraalgo(SingleSite(),cbealgo,truncscheme,Nsweep,tol)
    Alg = XTRGalgo(SingleSite(),algo,N,H)
    lnZ = 2log(normalize!(obj))
    return XTRG!(obj,Alg,lnZ)
end

function XTRG2!(obj::DenseMPO{L},H::SparseMPO{L},N::Int64;kwargs...) where L
    truncscheme = get(kwargs,:trunc,notrunc())
    tol = get(kwargs,:tol,1e-12)
    Nsweep = get(kwargs,:Nsweep,20)
    algo = Algebraalgo(DoubleSite(),NoAlgorithm(),truncscheme,Nsweep,tol)
    Alg = XTRGalgo(DoubleSite(),algo,N,H)
    lnZ = 2log(normalize!(obj))
    return XTRG!(obj,Alg,lnZ)
end
