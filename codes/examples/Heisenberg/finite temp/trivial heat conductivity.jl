using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

# function setdefault!(Env::Environment{4})
#     if issparse(Env.layer[1]) && issparse(Env.layer[3])
#         ρ1 = Env.layer[2]
#         ρ2′ = Env.layer[4]
#         tmpL = (isometry(space(ρ2′.ts[1])[4]', space(ρ1.ts[1])[2]))
#         tmpR = (isometry(space(ρ1.ts[end])[3]', space(ρ2′.ts[end])[1]))
#         Env.envs[1] = SparseLeftEnvironmentTensor([tmpL;;])
#         Env.envs[end] = SparseRightEnvironmentTensor([tmpR;;])
#     end
# end

# function contract(EnvL::SparseLeftEnvironmentTensor{2}, Hup::SparseMPOTensor, t::DenseMPOTensor, Hdown::SparseMPOTensor, t′::AdjointMPOTensor, EnvR::SparseRightEnvironmentTensor{2})
#     tmp = 0
#     validinds = filter(x -> !isnothing(Hup.m[x[1],x[3]]) && !isnothing(Hdown.m[x[2],x[4]]), [(i,j,k,l) for i in 1:EnvL.D[1], j in 1:EnvL.D[2], k in 1:EnvR.D[1], l in 1:EnvR.D[2]][:])
#     Nthr = get_num_threads_julia()
#     if Nthr > 1
#         Lock = Threads.ReentrantLock()
#         counter = Threads.Atomic{Int64}(1)
#         Threads.@sync for _ in 1:Nthr
#             Threads.@spawn while true
#                 ct = Threads.atomic_add!(counter, 1)
#                 ct > length(validinds) && break
#                 i,j,k,l = validinds[ct]
#                 C = contract(EnvL.A[i,j],Hup.m[i,k], t ,Hdown.m[j,l], t′ ,EnvR.A[k,l])
#                 lock(Lock)
#                 try
#                     tmp += C
#                 catch
#                     rethrow()
#                 finally
#                     unlock(Lock)
#                 end
#             end
#         end
#     else
#         for (i,j,k,l) in validinds
#             tmp += contract(EnvL.A[i,j],Hup.m[i,k], t ,Hdown.m[j,l], t′ ,EnvR.A[k,l])
#         end
#     end
#     return tmp
# end

# function pushright(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvL::SparseLeftEnvironmentTensor{2}, site::Int64)
#     tmpEnvL = Array{Any}(nothing, Hup.D[site][2], Hdown.D[site][2])
#     validinds = filter(x -> !isnothing(Hup.ts[site].m[x[1],x[3]]) && !isnothing(Hdown.ts[site].m[x[2],x[4]]), [(i,j,k,l) for i in 1:EnvL.D[1], j in 1:EnvL.D[2], k in 1:Hup.D[site][2], l in 1:Hdown.D[site][2]][:])
#     Nthr = get_num_threads_julia()
#     if Nthr > 1
#         Lock = Threads.ReentrantLock()
#         counter = Threads.Atomic{Int64}(1)
#         Threads.@sync for _ in 1:Nthr
#             Threads.@spawn while true
#                 ct = Threads.atomic_add!(counter, 1)
#                 ct > length(validinds) && break
#                 i,j,k,l = validinds[ct]
#                 C = pushright(Hup.ts[site].m[i,k],ρ.ts[site],Hdown.ts[site].m[j,l],ρ′.ts[site],EnvL.A[i,j])
#                 lock(Lock)
#                 try
#                     tmpEnvL[k,l] = axpy!(1,C,tmpEnvL[k,l])
#                     # sleep(1e-8)
#                 catch
#                     rethrow()
#                 finally
#                     unlock(Lock)
#                 end
#             end
#         end
#     else
#         for (i,j,k,l) in validinds
#             tmpEnvL[k,l] = axpy!(1,pushright(Hup.ts[site].m[i,k], ρ.ts[site], Hdown.ts[site].m[j,l], ρ′.ts[site], EnvL.A[i,j]),tmpEnvL[k,l])
#         end
#     end
#     return SparseLeftEnvironmentTensor(convert(Array{LeftEnvironmentTensor}, tmpEnvL))
# end

# function pushleft(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvR::SparseRightEnvironmentTensor{2}, site::Int64)
#     tmpEnvR = Array{Any}(nothing, Hup.D[site][1], Hdown.D[site][1])
#     validinds = filter(x -> !isnothing(Hup.ts[site].m[x[1],x[3]]) && !isnothing(Hdown.ts[site].m[x[2],x[4]]), [(i,j,k,l) for i in 1:Hup.D[site][1], j in 1:Hdown.D[site][1], k in 1:EnvR.D[1], l in 1:EnvR.D[2]][:])
#     Nthr = get_num_threads_julia()
#     if Nthr > 1
#         Lock = Threads.ReentrantLock()
#         counter = Threads.Atomic{Int64}(1)
#         Threads.@sync for _ in 1:Nthr
#             Threads.@spawn while true
#                 ct = Threads.atomic_add!(counter, 1)
#                 ct > length(validinds) && break
#                 i,j,k,l = validinds[ct]
#                 C = pushleft(Hup.ts[site].m[i,k],ρ.ts[site],Hdown.ts[site].m[j,l],ρ′.ts[site],EnvR.A[k,l])
#                 lock(Lock)
#                 try
#                     tmpEnvR[i,j] = axpy!(1,C,tmpEnvR[i,j])
#                     # sleep(1e-8)
#                 catch
#                     rethrow()
#                 finally
#                     unlock(Lock)
#                 end
#             end
#         end
#     else
#         for (i,j,k,l) in validinds
#             tmpEnvR[i,j] = axpy!(1,pushleft(Hup.ts[site].m[i,k],ρ.ts[site],Hdown.ts[site].m[j,l],ρ′.ts[site],EnvR.A[k,l]),tmpEnvR[i,j])
#         end
#     end
#     return SparseRightEnvironmentTensor(convert(Array{RightEnvironmentTensor}, tmpEnvR))
# end

# function contract(EnvL::LeftEnvironmentTensor{2},ht::LocalOperator{1,1},t::DenseMPOTensor{4},hb::LocalOperator{1,1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[4,3] * ht.A[1,5] * t.A[2,3,7,1] * hb.A[6,2] * t′.A[8,5,6,4] * EnvR.A[7,8]
# end

# function contract(EnvL::LeftEnvironmentTensor{2},::IdentityOperator{1},t::DenseMPOTensor{4},hb::LocalOperator{1,1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[3,2] * t.A[1,2,6,4] * hb.A[5,1] * t′.A[7,4,5,3] * EnvR.A[6,7]
# end

# function contract(EnvL::LeftEnvironmentTensor{2},ht::LocalOperator{1,1},t::DenseMPOTensor{4},::IdentityOperator{1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[3,2] * ht.A[1,4] * t.A[5,2,6,1] * t′.A[7,4,5,3] * EnvR.A[6,7]
# end

# function contract(EnvL::LeftEnvironmentTensor{2},::IdentityOperator{1},t::DenseMPOTensor{4},::IdentityOperator{1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[2,1] * t.A[3,1,5,4] * t′.A[6,4,3,2] * EnvR.A[5,6]
# end

# pushright(::Nothing, A::DenseMPOTensor{4}, h::AbstractLocalOperator, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2}) = contract(A,h,A′,EnvL)
# function pushright(h::LocalOperator{1, 1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ h.A[1,5] * A.A[4,2,-2,1] * A′.A[-1,5,4,3] * EnvL.A[3,2]
#     return LeftEnvironmentTensor(tmp)
# end
# function pushright(::IdentityOperator{1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[4,1,-2,3] * A′.A[-1,3,4,2] * EnvL.A[2,1]
#     return LeftEnvironmentTensor(tmp)
# end
# pushleft(::Nothing, A::DenseMPOTensor{4}, h::AbstractLocalOperator, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2}) = contract(A,h,A′,EnvR)
# function pushleft(h::LocalOperator{1, 1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ h.A[1,4] * A.A[5,-1,2,1] * A′.A[3,4,5,-2] * EnvR.A[2,3]
#     return RightEnvironmentTensor(tmp)
# end
# function pushleft(::IdentityOperator{1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[4,-1,1,3] * A′.A[2,3,4,-2] * EnvR.A[1,2]
#     return RightEnvironmentTensor(tmp)
# end
# ##
# function pushleft(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor x[-1;-2] ≔ ht.A[1,6] * objt.A[2,-1,3,1] * hb.A[5,2] * objb.A[4,6,5,-2] * EnvR.A[3,4]
#     return RightEnvironmentTensor(x)
# end

# function pushleft(::IdentityOperator{1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     # @tensor x[-1;-2] ≔ objt.A[2,-1,3,1] * hb.A[5,2] * objb.A[4,1,5,-2] * EnvR.A[3,4]
#     @tensor x[-1;-2] ≔ objt.A[1,-1,2,4] * hb.A[5,1] * objb.A[3,4,5,-2] * EnvR.A[2,3]
#     return RightEnvironmentTensor(x)
# end

# function pushleft(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     # @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[2,-1,3,1] * objb.A[4,5,2,-2] * EnvR.A[3,4]
#     @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[4,-1,2,1] * objb.A[3,5,4,-2] * EnvR.A[2,3]
#     return RightEnvironmentTensor(x)
# end

# function pushleft(::IdentityOperator{1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor x[-1;-2] ≔ objt.A[3,-1,1,4] * objb.A[2,4,3,-2] * EnvR.A[1,2]
#     return RightEnvironmentTensor(x)
# end


D = 64
Lx = 20
Ly = 1
params = (J = 1, Δ = 1.0)

# Latt = YCSqua(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

JE = let LocalSpace = TrivialSpinOneHalf, Root = InteractionTreeNode(), J=params.J, Δ = params.Δ
    # je = -1im*J^2/2
    # for i in 1:size(Latt)-2
    #     addIntr!(Root,(LocalSpace.S₊, LocalSpace.Sz, LocalSpace.S₋),(i,i+1,i+2),("S₊","Sz","S₋"),(false,false,false),je,nothing)
    #     addIntr!(Root,(LocalSpace.S₋, LocalSpace.Sz, LocalSpace.S₊),(i,i+1,i+2),("S₋","Sz","S₊"),(false,false,false),-je,nothing)
    #     addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₊, LocalSpace.S₋),(i,i+1,i+2),("Sz","S₊","S₋"),(false,false,false),-je*Δ,nothing)
    #     addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₋, LocalSpace.S₊),(i,i+1,i+2),("Sz","S₋","S₊"),(false,false,false),je*Δ,nothing)
    #     addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋, LocalSpace.Sz),(i,i+1,i+2),("S₊","S₋","Sz"),(false,false,false),-je*Δ,nothing)
    #     addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊, LocalSpace.Sz),(i,i+1,i+2),("S₋","S₊","Sz"),(false,false,false),je*Δ,nothing)
    # end
    je = -1im*J^2
    for i in 1:size(Latt)-2
        addIntr!(Root,(LocalSpace.S₊, LocalSpace.Sz, LocalSpace.S₋),(i,i+1,i+2),("S₊","Sz","S₋"),(false,false,false),   je*1,nothing)
        addIntr!(Root,(LocalSpace.S₋, LocalSpace.Sz, LocalSpace.S₊),(i,i+1,i+2),("S₋","Sz","S₊"),(false,false,false),  -je*1,nothing)
        addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₊, LocalSpace.S₋),(i,i+1,i+2),("Sz","S₊","S₋"),(false,false,false),-Δ*je*1,nothing)
        addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₋, LocalSpace.S₊),(i,i+1,i+2),("Sz","S₋","S₊"),(false,false,false), Δ*je*1,nothing)
        addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋, LocalSpace.Sz),(i,i+1,i+2),("S₊","S₋","Sz"),(false,false,false),-Δ*je*1,nothing)
        addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊, LocalSpace.Sz),(i,i+1,i+2),("S₋","S₊","Sz"),(false,false,false), Δ*je*1,nothing)
    end

    # edgepairs = [(size(Latt),1,2), (size(Latt)-1,size(Latt),1)]

    # addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₋, LocalSpace.S₊),(1,2,size(Latt)),("Sz","S₋","S₊"),(false,false,false),je,nothing)
    # addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₊, LocalSpace.S₋),(1,2,size(Latt)),("Sz","S₊","S₋"),(false,false,false),-je,nothing)
    # addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋, LocalSpace.Sz),(1,2,size(Latt)),("S₊","S₋","Sz"),(false,false,false),-je*Δ,nothing)
    # addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊, LocalSpace.Sz),(1,2,size(Latt)),("S₋","S₊","Sz"),(false,false,false),je*Δ,nothing)
    # addIntr!(Root,(LocalSpace.S₋, LocalSpace.Sz, LocalSpace.S₊),(1,2,size(Latt)),("S₋","Sz","S₊"),(false,false,false),-je*Δ,nothing)
    # addIntr!(Root,(LocalSpace.S₊, LocalSpace.Sz, LocalSpace.S₋),(1,2,size(Latt)),("S₊","Sz","S₋"),(false,false,false),je*Δ,nothing)

    # addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊, LocalSpace.Sz),(1,size(Latt)-1,size(Latt)),("S₋","S₊","Sz"),(false,false,false),je,nothing)
    # addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋, LocalSpace.Sz),(1,size(Latt)-1,size(Latt)),("S₊","S₋","Sz"),(false,false,false),-je,nothing)
    # addIntr!(Root,(LocalSpace.S₋, LocalSpace.Sz, LocalSpace.S₊),(1,size(Latt)-1,size(Latt)),("S₋","Sz","S₊"),(false,false,false),-je*Δ,nothing)
    # addIntr!(Root,(LocalSpace.S₊, LocalSpace.Sz, LocalSpace.S₋),(1,size(Latt)-1,size(Latt)),("S₊","Sz","S₋"),(false,false,false),je*Δ,nothing)
    # addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₊, LocalSpace.S₋),(1,size(Latt)-1,size(Latt)),("Sz","S₊","S₋"),(false,false,false),-je*Δ,nothing)
    # addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₋, LocalSpace.S₊),(1,size(Latt)-1,size(Latt)),("Sz","S₋","S₊"),(false,false,false),je*Δ,nothing)
    AutomataSparseMPO(Root,size(Latt))
    # Root
end



@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
lsβ2 = 2lsβ[2:end]
lsI = zeros(length(lsρ))
for (i,ρ) in enumerate(lsρ)
    @show i
    Env = Environment([JE,ρ,JE,ρ'])
    initialize!(Env)
    lsI[i] = _scalar(Env) / size(Latt)
    # @show lsI[i]
end
# @show length(lsβ),length(lsρ)
@save "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI

@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2


