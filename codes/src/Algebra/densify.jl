# function densify!(α::Number, x::SparseMPO, y::DenseMPO;kwargs...)
#     trunc = get(kwargs,:trunc,notrunc())
#     N  = get(kwargs,:N,3)
#     tol = get(kwargs,:tol,1e-8)
#     algo = Algebraalgo(DoubleSite(),NoAlgorithm(),trunc,N,tol)
#     return densify!(α,x,y,algo;kwargs...)
# end

# function densify!(α::Number, x::SparseMPO{L}, y::DenseMPO{L}, Alg::Algebraalgo;kwargs...) where L
#     y′ = y'
    
#     to = TimerOutput()
#     @timeit to "initialize XY Env" begin
#         Env = Environment([deepcopy(x),y′])
#         initialize!(Env;
#         left_default_space = ntuple(x -> space(DeS₀.ts[1])[2],2),
#         right_default_space = ntuple(x -> space(DeS₀.ts[end])[3]',2))
#     end

#     info = Algebrainfo()
#     while info.n ≤ Alg.N
#         localto = TimerOutput()

#         l2rinfo = Algebrasweepinfo(L2R())
#         mto = densify!(α,Env,Alg,l2rinfo)
#         merge!(localto,mto)
#         merge!(info,l2rinfo)

#         r2linfo = Algebrasweepinfo(R2L())
#         mto = densify!(α,Env,Alg,r2linfo)
#         merge!(localto,mto)
#         merge!(info,r2linfo)

#         show(localto;title = "densify!")
#         print("\n")
#         show(info)
#         merge!(to,localto)

#         info.err < Alg.tol && break
#     end

#     return xp!(y′',y),to,info
# end

# function densify!(α::Number, Env::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{L2R})
#     localto = TimerOutput()
#     L = length(Env.layer[2])
#     for site in 1:L-1
#         siteinfo = Algebrasiteinfo()
#         @timeit localto "composite x₀" x₀ = deepcopy(composite(Env.layer[2].ts[site:site+1]...))
#         @show space(Env.layer[2].ts[site])
#         @assert (x2 = norm(x₀)^2) ≠ 0
#         @timeit localto "projection" projH = proj2(Env,site,site+1)
#         @timeit localto "SVD" tl, tc, tr, ~ = tsvd(rmul!(action(projH, projH.H),α); direction=:center,trunc = Alg.trunc)
#         siteinfo.bond = BondInfo(tc)
#         @timeit localto "contract" tr = contract(tc,tr)
#         Env.layer[2].ts[site:site+1] = adjoint.([tl, tr])
#         canonicalize!(Env.layer[2],site+1)
#         @timeit localto "push right" pushright!(Env)
#         @timeit localto "composite x" x = composite(Env.layer[2].ts[site:site+1]...)
#         siteinfo.err = norm(x-x₀)^2/x2
#         merge!(sweepinfo,siteinfo)
#     end

#     return localto
# end

# function densify!(α::Number, Env::Environment{2}, Alg::Algebraalgo{DoubleSite}, sweepinfo::Algebrasweepinfo{R2L})
#     localto = TimerOutput()
#     L = length(Env.layer[2])
#     for site in L:-1:2
#         siteinfo = Algebrasiteinfo()
#         @timeit localto "composite x₀" x₀ = deepcopy(composite(Env.layer[2].ts[site-1:site]...))
#         @assert (x2 = norm(x₀)^2) ≠ 0
#         @timeit localto "projection" projH = proj2(Env,site-1,site)
#         @timeit localto "SVD" tl, tc, tr, ~ = tsvd(rmul!(action(projH, projH.H),α); direction=:center,trunc = Alg.trunc)
#         siteinfo.bond = BondInfo(tc)
#         @timeit localto "contract" tl = contract(tl,tc) 
#         Env.layer[2].ts[site-1:site] = adjoint.([tl, tr])
#         canonicalize!(Env.layer[2],site-1)
#         @timeit localto "push left" pushleft!(Env)
#         @timeit localto "composite x" x = composite(Env.layer[2].ts[site-1:site]...)
#         siteinfo.err = norm(x-x₀)^2/x2
#         merge!(sweepinfo,siteinfo)
#     end

#     return localto
# end


# function contract(El::LeftEnvironmentTensor{2}, A::LocalOperator{1, 2})
#     iso = _isometry(space(El)[2]',space(A)[2]')'
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,1] * A.A[-2,2,-4] * iso[1,2,-3]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftEnvironmentTensor{2}, A::LocalOperator{2, 1})
#     iso = _isometry(space(El)[2]',space(A)[2]')'
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,1] * A.A[-2,2,-4] * iso[1,2,-3]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftEnvironmentTensor{2}, A::LocalOperator{1, 1})
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,-3] * A.A[-2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::IdentityOperator{1})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,-4,-6] * A.A[-3,-5]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftEnvironmentTensor{2}, A::IdentityOperator{1})
#     @tensor tmp[-1,-2;-3,-4] ≔ El.A[-1,-3] * A.A[-2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::LocalOperator{1, 2})
#     iso = _isometry(space(El)[3]',space(A)[2]')'
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,1,-6] * A.A[-3,2,-5] * iso[1,2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::LocalOperator{2, 1})
#     iso = _isometry(space(El)[3]',space(A)[2]')'
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,1,-6] * A.A[-3,2,-5] * iso[1,2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 3}, A::LocalOperator{1, 1})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-1,-2,-4,-6] * A.A[-3,-5]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(::IdentityOperator{1}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ obj.A[2,1,1,-2] * EnvR.A[-1,2]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 2}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     iso = _isometry(space(A)[2],space(EnvR)[1])
#     @tensor tmp[-1;-2] ≔ iso[-1,4,5] * A.A[2,4,1] * obj.A[3,1,2,-2] * EnvR.A[5,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{2, 1}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     iso = _isometry(space(A)[2],space(EnvR)[1])
#     @tensor tmp[-1;-2] ≔ iso[-1,4,5] * A.A[2,4,1] * obj.A[3,1,2,-2] * EnvR.A[5,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 1}, obj::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[2,1] * obj.A[3,1,2,-2] * EnvR.A[-1,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 2}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     iso = _isometry(space(EnvL)[2]',space(A)[2]')'
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,4] * A.A[2,5,1] * obj.A[-1,1,2,3] * iso[4,5,-2]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{2, 1}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     iso = _isometry(space(EnvL)[2]',space(A)[2]')'
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,4] * A.A[2,5,1] * obj.A[-1,1,2,3] * iso[4,5,-2]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::LocalOperator{1, 1}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,-2] * A.A[2,1] * obj.A[-1,1,2,3]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(::IdentityOperator{1}, obj::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ EnvL.A[2,-2] * obj.A[-1,1,1,2]
#     return LeftEnvironmentTensor(tmp)
# end

function pushleft(A::AbstractMPS, mpo::DenseMPO, B::AbstractMPS, EnvR::DenseRightEnvironmentTensor{3}, site::Int64)
    x = contract(A.ts[site], mpo.ts[site], B.ts[site], EnvR.A)
    return DenseRightEnvironmentTensor(x)
end


function pushright(A::AbstractMPS, mpo::DenseMPO, B::AbstractMPS, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64)
    x = contract(A.ts[site], mpo.ts[site], B.ts[site], EnvL.A)
    return DenseLeftEnvironmentTensor(x)
end

function contract(A::MPSTensor{3}, mpot::DenseMPOTensor{4}, B::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1,-2;-3] ≔ A.A[-1,2,1] * mpot.A[5,-2,3,2] * B.A[4,-3,5] * EnvR.A[1,3,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, mpot::DenseMPOTensor{4}, B::AdjointMPSTensor{3}, EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2 -3] ≔ A.A[5,4,-3] * mpot.A[2,3,-2,4] * B.A[-1,1,2] * EnvL.A[1,3,5]
    return LeftEnvironmentTensor(tmp)
end

function action(O::DenseProjectiveHamiltonian{3,1}, obj::MPSTensor{3})
    h = O.H[1]
    @tensor x[-1 -2;-3] ≔ O.EnvL.A.A[-1,3,1] * obj.A[1,2,4] * O.EnvR.A.A[4,5,-3] * h.A[-2,3,5,2]
    x = MPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function action(O::DenseProjectiveHamiltonian{3,2}, obj::CompositeMPSTensor{2,4})
    hl,hr = O.H
    @tensor x[-1 -2 -3;-4] ≔ O.EnvL.A.A[-1,3,1] * obj.A[1,2,4,6] * O.EnvR.A.A[6,7,-4] * hl.A[-2,3,5,2] * hr.A[-3,5,7,4]
    x = CompositeMPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end


function densify!(ρ::DenseMPO,H::SparseMPO;kwargs...)
    trunc = get(kwargs,:trunc,notrunc())
    algo = get(kwargs,:algo,CBEalgo(dynamicSVD(1.2,2),NoStruc(),0,_getdim(trunc)))
    tol = get(kwargs,:tol,1e-12)
    return mul!(ρ,ρ,H,1,Algebraalgo(SingleSite(),algo,trunc,3,tol))[1]
end

function orthogonalize!(H::DenseMPOTensor,A::T,A′::T,EnvL::DenseLeftEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    C = contract(EnvL.A,A,H) |> x -> x - contract(x,A′)
    return C
end

function orthogonalize!(H::DenseMPOTensor,B::T,B′::T,EnvR::DenseRightEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    C = contract(B,H,EnvR.A) |> x -> x - contract(x,B′)
    return C
end

function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{4})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,3,1] * A.A[1,2,-4] * mpo.A[-2,3,-3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, B::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[-1,3,1] * B.A[-3,-2,2,3] * EnvR.A[1,2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{2,4}, B::MPSTensor{3})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvR.A[-1,-2,2,1] * B'.A[1,3,2] * B.A[3,-3,-4])
end

function contract(EnvR::RightCompositeEnvironmentTensor{2,4}, A::AdjointMPSTensor{3})
    @tensor tmp[-1 -2;-3] ≔ EnvR.A[-1,-2,2,1] * A.A[1,-3,2] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{2, 4})
    @tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,2,1] * EnvR.A[1,2,-2,-3]
    return MPSTensor(tmp)
end

function contract(EnvL::RightCompositeEnvironmentTensor{2, 4}, Λ::MPSTensor{2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])
end

function action(O::DenseProjectiveHamiltonian{3,0}, obj::MPSTensor{2})
    @tensor x[-1;-2] ≔ O.EnvL.A.A[-1,3,1] * obj.A[1,2] * O.EnvR.A.A[2,3,-2]
    x = MPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end