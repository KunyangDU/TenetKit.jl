

function SETTN!(β::Number, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L
    
    N = get(kwargs,:max_order,10)
    D = get(kwargs,:D,max(maximum(vcat(collect.(H.D)...)),_maxdim(ρ)))
    tol = get(kwargs,:tol,1e-12)

    alg = SETTNalgo(SingleSite(),N,D,tol)
    return SETTN!(β, H, ρ, alg)
end

function SETTN!(β::Number,H::SparseMPO{L}, ρ::DenseMPO, Alg::SETTNalgo{SingleSite}) where L

    to = TimerOutput()

    @timeit to "I - βH" begin
        Hn = deepcopy(ρ)
        F₀,localto = SETTN!(β,H,Hn,ρ,1,Alg)
    end
    merge!(to,localto, tree_point = ["I - βH"])
    show(localto;title = "SETTN - (I - βH)")
    print("\n")
    
    for i in 2:Alg.N 
        @timeit to "Iteration" F,localto = SETTN!(β,H,Hn,ρ,i,Alg)

        ϵ = abs((F - F₀) / F)
        merge!(to,localto, tree_point = ["Iteration"])   
        show(localto;title = "SETTN - $(i) (≤$(Alg.N))")
        println("\ndF = $(ϵ)")
        if ϵ < Alg.tol
            println("SETTN converged at $(i)th order with dF = $(ϵ)")
            break
        end
             
        i == Alg.N && println("SETTN not converged at max $(i)th order with dF = $(ϵ)") 
        F₀ = F
    end

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


# function SETTN!(lsβ::Vector, H::SparseMPO{L}, ρ::DenseMPO;kwargs...) where L
    
#     max_order = get(kwargs,:max_order,6)
#     D = get(kwargs,:D,maximum(vcat(collect.(H.D)...)))
#     F_tol = get(kwargs,:F_tol,1e-8)
#     F = zeros(max_order)
#     lsρ = Vector(undef,length(lsβ))
#     for i in 1:length(lsβ)-1
#         lsρ[i] = deepcopy(ρ)
#     end

#     β = lsβ[end]

#     Hn = deepcopy(ρ)
#     dF = 2*F_tol # make sure dF > F_tol
#     for i in 1:max_order 
#         println("SETTN order = $i")
#         mul!(Hn,deepcopy(Hn),H,1.,0.; D = D)
#         axpy!((-β)^i / factorial(i),Hn ,ρ ; D = D)

#         F[i] = - log(tr(ρ)) /2/β
#         if i ≠ 1
#             dF = abs((F[i] - F[i-1]) / F[i])
#             println("dF = $dF")
#         end
#         for (iβ,βi) in enumerate(lsβ[1:end-1])
#             lsρ[iβ] = axpy!((-βi)^i / factorial(i),Hn ,lsρ[iβ] ; D = D)
#         end
        

#         if dF < F_tol
#            println("SETTN converged at $(i)th order with dF = $(dF)")
#            break
#         end

#         i == max_order && println("SETTN not converged at max $(i)th order with dF = $(dF)") 
#     end
#     lsρ[end] = deepcopy(ρ)

#     return lsρ
# end

