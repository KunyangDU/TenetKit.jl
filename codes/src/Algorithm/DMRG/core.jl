

function DMRG!(Env::Environment{3,L}, Alg::DMRGalgo;kwargs...) where L
    lsinfo = []
    lsE = []
    Sv = nothing
    __init_io__()
    for i in 1:Alg.N
        info =  DMRGinfo()
        l2rinfo = DMRGsweepinfo(L2R())
        to = DMRG!(Env,Alg,l2rinfo)
        _merge_io!(to)
        show(to;title=">>> DMRG ($(i)/$(Alg.N)) >>>")
        print("\n")
        show(l2rinfo)
        merge!(info,l2rinfo)
        flush(stdout)

        r2linfo = DMRGsweepinfo(R2L())
        to = DMRG!(Env,Alg,r2linfo)
        _merge_io!(to)
        show(to;title="<<< DMRG ($(i)/$(Alg.N)) <<<")
        print("\n")
        show(r2linfo)
        merge!(info,r2linfo)
        ΔE = (r2linfo.E[end] - l2rinfo.E[end]) / (r2linfo.E[end] + l2rinfo.E[end])
        # ΔS = norm((l2rinfo.S .- reverse(r2linfo.S))) / norm((l2rinfo.S .+ reverse(r2linfo.S))/2)
        ΔS = abs(maximum(l2rinfo.S) - maximum(r2linfo.S))/(maximum(l2rinfo.S) + maximum(r2linfo.S))
        println("ΔE = El - Er = $(ΔE), ΔS = |ΔŜ|/|Ŝ| = $(ΔS)")
        flush(stdout)

        push!(lsinfo,deepcopy(info))
        push!(lsE,info.E[end])

        if ΔE < Alg.Etol && ΔS < Alg.Stol
            return lsE,lsinfo
        end
    end
    return lsE,lsinfo
end
