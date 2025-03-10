

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
    Lanczosinfo(info::KrylovKit.ConvergenceInfo) = new(info.converged, info.numiter)
    Lanczosinfo() = new(1,0)
end

mutable struct DMRGinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    n::Int64
    ϵ::Number
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
    ϵ::Number
    E::Number
    σE::Number
    DMRGsweepinfo(direction::SweepDirection, bond::BondInfo, solver::SolverInfo, ϵ::Number,E::Number, σE::Number) = new{typeof(direction)}(direction,bond,solver,ϵ,E,σE)
    DMRGsweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(),0,Inf,0)
end

mutable struct DMRGsiteinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    ϵ::Number
    E::Number
    σE::Number
    DMRGsiteinfo(bond::BondInfo, solver::SolverInfo, ϵ::Number,E::Number, σE::Number) = new(bond,solver,ϵ,E,σE)
    DMRGsiteinfo() = new(BondInfo(), Lanczosinfo(),0,Inf,0)
end

mutable struct CBEinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    ϵ::Number
    CBEinfo(direction::SweepDirection,ϵ::Number) = new{typeof(direction)}(direction,ϵ)
    CBEinfo(direction::SweepDirection) = new{typeof(direction)}(direction,0)
end

mutable struct TDVPinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    n::Int64
    ϵ::Number
    TDVPinfo(bond::BondInfo, solver::SolverInfo,n::Int64,ϵ::Number) = new(bond,solver,n,ϵ)
    TDVPinfo(info::TDVPinfo) = new(BondInfo(),Lanczosinfo(),info.n,0)
    TDVPinfo() = new(BondInfo(), Lanczosinfo(),0,0,Inf,0)
end

mutable struct TDVPsweepinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    bond::BondInfo
    solver::SolverInfo
    ϵ::Number

    TDVPsweepinfo(direction::SweepDirection, bond::BondInfo, solver::SolverInfo, ϵ::Number) = new{typeof(direction)}(direction,bond,solver,ϵ)
    TDVPsweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(),0)
end

mutable struct TDVPsiteinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    ϵ::Number
    TDVPsiteinfo(bond::BondInfo, solver::SolverInfo, ϵ::Number) = new(bond,solver,ϵ)
    TDVPsiteinfo() = new(BondInfo(), Lanczosinfo(),0)
end

# function merge(A::DMRGsweepinfo{dir₁},B::DMRGsweepinfo{dir₂}) where {dir₁,dir₂}
#     @assert dir₁ == dir₂ "direction mismatch"
#     return DMRGsweepinfo(sch₁,merge(A.bond, B.bond),merge(A.solver, B.solver),min(A.E,B.E),max(A.σE,B.σE))
# end

function merge!(A::DMRGinfo,B::DMRGsweepinfo{dir}) where dir
    merge!(A.bond, B.bond)
    merge!(A.solver, B.solver)
    A.ϵ = max(A.ϵ,B.ϵ)
    A.E = min(A.E,B.E)
    A.σE = max(A.σE,B.σE)
    dir <: R2L && (A.n += 1)
    return A
end

function merge!(A::DMRGsweepinfo{dir},B::DMRGsiteinfo) where dir
    merge!(A.bond, B.bond)
    merge!(A.solver, B.solver)
    A.ϵ = max(A.ϵ,B.ϵ)
    A.E = min(A.E,B.E)
    A.σE = max(A.σE,B.σE)
    return A
end

function merge!(info1::DMRGsiteinfo, info2::CBEinfo)
    info1.ϵ = max(info1.ϵ, info2.ϵ)
end

#= ========================= =#

function merge!(A::Lanczosinfo,B::Lanczosinfo)
    A.converged = A.converged & B.converged
    A.numiter = max(A.numiter,B.numiter)
    return A
end

#= ========================= =#

function merge!(A::BondInfo,B::BondInfo)
    A.Deff = max(A.Deff, B.Deff)
    A.D = max(A.D,B.D)
    A.S = max(A.S,B.S)
    return A
end

merge(A::BondInfo,B::BondInfo) = BondInfo(max(A.Deff, B.Deff), max(A.D,B.D), max(A.S,B.S) )

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



