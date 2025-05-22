

function SETTN!(β::Number, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L
    
    N = get(kwargs,:max_order,10)
    D = get(kwargs,:D,max(maximum(vcat(collect.(H.D)...)),_maxdim(ρ)))
    tol = get(kwargs,:tol,1e-12)

    alg = SETTNalgo(SingleSite(),N,D,tol)
    return SETTN!(β, H, ρ, alg)
end

function SETTN!(β::Number,H::SparseMPO{L}, ρ::DenseMPO, Alg::SETTNalgo{SingleSite}) where L

    to = TimerOutput()
    # H2 = nothing

    @timeit to "I - βH" begin
        Hn = deepcopy(ρ)
        F₀,localto = SETTN!(β,H,Hn,ρ,1,Alg)
    end
    merge!(to,localto, tree_point = ["I - βH"])
    show(localto;title = "SETTN - (I - βH)")
    print("\n")
    flush(stdout)
    
    for i in 2:Alg.N 
        @timeit to "Iteration" F,localto = SETTN!(β,H,Hn,ρ,i,Alg)

        ϵ = abs((F - F₀) / F)
        merge!(to,localto, tree_point = ["Iteration"])   
        show(localto;title = "SETTN - $(i) (≤$(Alg.N))")
        println("\ndF = $(ϵ)")
        flush(stdout)
        if ϵ < Alg.tol
            println("SETTN converged at $(i)th order with dF = $(ϵ)")
            break
        end
             
        i == Alg.N && println("SETTN not converged at max $(i)th order with dF = $(ϵ)") 
        # i == 2 && (H2 = deepcopy(Hn))
        F₀ = F

        GC.gc()
    end
    flush(stdout)

    show(to;title = "SETTN")
    print("\n")

    return ρ
    
end

function SETTN!(β::Number,H::SparseMPO{L},Hn::DenseMPO,ρ::DenseMPO,order::Int64,Alg::SETTNalgo{SingleSite}) where L
    to = TimerOutput()
    @timeit to "mul!" ~,multo = mul!(Hn,deepcopy(Hn),H; D = Alg.D)
    @timeit to "axpy!" ~,axpyto = axpy!((-β) ^ order / factorial(order),Hn ,ρ ; D = Alg.D)
    merge!(to,multo, tree_point = ["mul!"])
    merge!(to,axpyto, tree_point = ["axpy!"])
    @timeit to "calculate F" F = - log(tr(ρ)) / 2 / β
    return F,to
end



