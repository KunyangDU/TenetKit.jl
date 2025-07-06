

mutable struct BondInfo <: AbstractInformation
    Deff::Int64 
    D::Int64
    S::Number
    BondInfo(Deff::Int64,D::Int64,S::Number) = new(Deff,D,S)   
    BondInfo() = new(0,0,0)
end

mutable struct Lanczosinfo <: SolverInfo
    converged::Int
    numiter::Int
    Lanczosinfo(converged::Int, numiter::Int) = new(converged,numiter)
    Lanczosinfo(info::KrylovKit.ConvergenceInfo) = new(info.converged, info.numops)
    Lanczosinfo() = new(1,0)
end

mutable struct DMRGinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    n::Int64
    err::Number
    E::Number
    σE::Number
    DMRGinfo(bond::BondInfo, solver::SolverInfo,n::Int64,ϵ::Number, E::Number, σE::Number) = new(bond,solver,n,ϵ,E,σE)
    DMRGinfo(info::DMRGinfo) = new(BondInfo(),Lanczosinfo(),info.n,0,info.E,info.σE)
    DMRGinfo() = new(BondInfo(), Lanczosinfo(),0,0,Inf,0)
end

mutable struct DMRGsweepinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    bond::BondInfo
    solver::SolverInfo
    err::Number
    E::Number
    σE::Number
    DMRGsweepinfo(direction::SweepDirection, bond::BondInfo, solver::SolverInfo, ϵ::Number,E::Number, σE::Number) = new{typeof(direction)}(direction,bond,solver,ϵ,E,σE)
    DMRGsweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(),0,Inf,0)
end

mutable struct DMRGsiteinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    err::Number
    E::Number
    σE::Number
    DMRGsiteinfo(bond::BondInfo, solver::SolverInfo, ϵ::Number,E::Number, σE::Number) = new(bond,solver,ϵ,E,σE)
    DMRGsiteinfo() = new(BondInfo(), Lanczosinfo(),0,Inf,0)
end

mutable struct CBEinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    err::Number
    CBEinfo(direction::SweepDirection,ϵ::Number) = new{typeof(direction)}(direction,ϵ)
    CBEinfo(direction::SweepDirection) = new{typeof(direction)}(direction,0)
end

mutable struct TDVPinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    n::Int64
    err::Number
    lnZ::Number
    E::Number
    TDVPinfo(bond::BondInfo, solver::SolverInfo,n::Int64,ϵ::Number,lnZ::Number,E::Number) = new(bond,solver,n,ϵ,lnZ,E)
    TDVPinfo(info::TDVPinfo) = new(BondInfo(),Lanczosinfo(),info.n,0,info.lnZ,info.E)
    TDVPinfo() = new(BondInfo(), Lanczosinfo(),0,0,0,0)
    TDVPinfo(lnZ::Number) = new(BondInfo(), Lanczosinfo(),0,0,lnZ,0)
end

mutable struct TDVPsweepinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    bond::BondInfo
    solver::SolverInfo
    err::Number
    TDVPsweepinfo(direction::SweepDirection, bond::BondInfo, solver::SolverInfo, ϵ::Number) = new{typeof(direction)}(direction,bond,solver,ϵ)
    TDVPsweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(),0)
    TDVPsweepinfo(direction::SweepDirection,err::Number) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(),err)
end

mutable struct TDVPsiteinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    err::Number
    TDVPsiteinfo(bond::BondInfo, solver::SolverInfo, ϵ::Number) = new(bond,solver,ϵ)
    TDVPsiteinfo() = new(BondInfo(), Lanczosinfo(),0)
end

mutable struct SETTNinfo <: AlgorithmInfo
    bond::BondInfo
    n::Int64
    err::Number
    lnZ::Number
    SETTNinfo(bond::BondInfo,n::Int64,ϵ::Number,lnZ::Number) = new(bond,n,ϵ,lnZ)
    SETTNinfo(info::SETTNinfo) = new(BondInfo(),info.n,NaN,NaN)
    SETTNinfo() = new(BondInfo(),0,NaN,NaN)
end

mutable struct SETTNsweepinfo <: AlgorithmInfo
    bond::BondInfo
    err::Number
    lnZ::Number
    SETTNsweepinfo(bond::BondInfo, ϵ::Number, lnZ::Number) = new(bond,ϵ,lnZ)
    SETTNsweepinfo(err::Number) = new(BondInfo(),err,0)
    SETTNsweepinfo() = new(BondInfo(),0,0)
end

mutable struct Algebrainfo <: AlgorithmInfo
    bond::BondInfo
    n::Int64
    err::Number
    Algebrainfo(bond::BondInfo, n::Int64, ϵ::Number) = new(bond,n,ϵ)
    Algebrainfo(info::Algebrainfo) = new(BondInfo(),info.n,0)
    Algebrainfo() = new(BondInfo(),0,0)
end

mutable struct Algebrasweepinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    bond::BondInfo
    err::Number
    Algebrasweepinfo(direction::SweepDirection, bond::BondInfo, ϵ::Number) = new{typeof(direction)}(direction, bond, ϵ)
    Algebrasweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), 0)
end

mutable struct Algebrasiteinfo <: AlgorithmInfo
    bond::BondInfo
    err::Number
    Algebrasiteinfo(bond::BondInfo, ϵ::Number) = new(bond,ϵ)
    Algebrasiteinfo() = new(BondInfo(),0)
end

# function merge(A::DMRGsweepinfo{dir₁},B::DMRGsweepinfo{dir₂}) where {dir₁,dir₂}
#     @assert dir₁ == dir₂ "direction mismatch"
#     return DMRGsweepinfo(sch₁,merge(A.bond, B.bond),merge(A.solver, B.solver),min(A.E,B.E),max(A.σE,B.σE))
# end

function TimerOutputs.merge!(A::DMRGinfo,B::DMRGsweepinfo{dir}) where dir
    merge!(A.bond, B.bond)
    merge!(A.solver, B.solver)
    A.err = max(A.err,B.err)
    A.E = min(A.E,B.E)
    A.σE = max(A.σE,B.σE)
    dir <: R2L && (A.n += 1)
    return A
end

function TimerOutputs.merge!(A::TDVPinfo,B::TDVPsweepinfo{dir}) where dir
    merge!(A.bond, B.bond)
    merge!(A.solver, B.solver)
    A.err = B.err
    return A
end

function TimerOutputs.merge!(A::Algebrainfo,B::Algebrasweepinfo{dir}) where dir
    merge!(A.bond, B.bond)
    A.err = B.err
    dir <: R2L && (A.n += 1)
    return A
end

function TimerOutputs.merge!(A::SETTNinfo,B::SETTNsweepinfo)
    merge!(A.bond, B.bond)
    A.lnZ = B.lnZ
    A.n += 1
    return A
end

function TimerOutputs.merge!(A::T₁,B::T₂) where {T₁<:Union{DMRGsweepinfo,TDVPsweepinfo},T₂<:Union{DMRGsiteinfo,TDVPsiteinfo}}
    merge!(A.bond, B.bond)
    merge!(A.solver, B.solver)
    
    if T₁ <: DMRGsweepinfo && T₂ <: DMRGsiteinfo
        A.err = max(A.err,B.err)
        A.E = min(A.E,B.E)
        A.σE = max(A.σE,B.σE)
    else
        A.err = A.err + B.err
    end

    return A
end

function TimerOutputs.merge!(A::Algebrasweepinfo,B::Algebrasiteinfo)
    merge!(A.bond, B.bond)
    A.err = max(A.err,B.err)
    return A
end

function TimerOutputs.merge!(info1::DMRGsiteinfo, info2::CBEinfo)
    info1.err = max(info1.err, info2.err)
end

function TimerOutputs.merge!(info1::TDVPsiteinfo, info2::CBEinfo)
    info1.err = info1.err + info2.err
end

TimerOutputs.merge!(::Algebrasiteinfo, ::CBEinfo) = nothing

#= ========================= =#

function TimerOutputs.merge!(A::Lanczosinfo,B::Lanczosinfo)
    A.converged = A.converged & B.converged
    A.numiter = max(A.numiter,B.numiter)
    return A
end

#= ========================= =#

function TimerOutputs.merge!(A::BondInfo,B::BondInfo)
    A.Deff = max(A.Deff, B.Deff)
    A.D = max(A.D,B.D)
    A.S = isnan(B.S) ? A.S : max(A.S,B.S)
    return A
end

TimerOutputs.merge(A::BondInfo,B::BondInfo) = BondInfo(max(A.Deff, B.Deff), max(A.D,B.D), max(A.S,B.S) )

function update!(A::BondInfo,B::Union{MPSTensor{2},DenseMPOTensor{2}})
    merge!(A,BondInfo(B))
end

BondInfo(A::AbstractTensorWrapper) = BondInfo(A.A)

function BondInfo(A::TensorMap{<:GradedSpace,1,1})
    bondinfo = BondInfo()
    for (c,b) in blocks(A)
        λ = diag(b)
        bondinfo.Deff += length(λ)
        bondinfo.D += length(λ) * dim(c)
    end
    bondinfo.S = vonNeumann(A)
    return bondinfo
end

function BondInfo(A::TensorMap{<:ComplexSpace,1,1})
    λ = diag(A.data)
    return BondInfo(length(λ),length(λ),vonNeumann(A))
end

function Base.show(io::IO,info::Algebrainfo)
    println(io,info.bond,", ProjErr = $(info.err)")
end

function Base.show(io::IO,info::SETTNinfo)
    println(io,info.bond,", lnZ = $(info.lnZ), lnZ Err = $(info.err)")
end

function Base.show(io::IO,info::DMRGsweepinfo)
    println(io,info.bond,", K = $(info.solver.numiter), TruncError = $(info.err), E = $(info.E), σE = $(info.σE)")
end

function Base.show(io::IO,info::TDVPsweepinfo)
    println(io,info.bond,", K = $(info.solver.numiter), TruncError = $(info.err)")
end

function Base.show(io::IO,info::SETTNsweepinfo)
    println(io,info.bond,", lnZ = $(info.lnZ), AlgebraErr = $(info.err)")
end

function Base.show(io::IO,info::Algebrasweepinfo)
    println(io,info.bond,", ProjErr = $(info.err)")
end

function Base.show(io::IO,info::BondInfo)
    print(io,"D( $(info.Deff) => $(info.D) ), vnS = $(info.S)")
end


