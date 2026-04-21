

mutable struct BondInfo <: AbstractInformation
    Deff::Int64
    D::Int64
    S::Float64
    BondInfo(Deff::Int64, D::Int64, S::Real) = new(Deff, D, Float64(S))
    BondInfo() = new(0, 0, 0.0)
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
    err::Float64
    E::Vector{Float64}
    S::Vector{Float64}
    DMRGinfo(bond::BondInfo, solver::SolverInfo, n::Int64, ϵ::Real, E::Vector{Float64}, S::Vector{Float64}) = new(bond, solver, n, Float64(ϵ), E, S)
    DMRGinfo(info::DMRGinfo) = new(BondInfo(), Lanczosinfo(), info.n, 0.0, info.E, info.S)
    DMRGinfo() = new(BondInfo(), Lanczosinfo(), 0, 0.0, Float64[], Float64[])
end

mutable struct DMRGsweepinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    bond::BondInfo
    solver::SolverInfo
    err::Float64
    E::Vector{Float64}
    S::Vector{Float64}
    DMRGsweepinfo(direction::SweepDirection, bond::BondInfo, solver::SolverInfo, ϵ::Real, E::Vector{Float64}, S::Vector{Float64}) = new{typeof(direction)}(direction, bond, solver, Float64(ϵ), E, S)
    DMRGsweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(), 0.0, Float64[], Float64[])
end

mutable struct DMRGsiteinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    err::Float64
    E::Float64
    S::Float64
    DMRGsiteinfo(bond::BondInfo, solver::SolverInfo, ϵ::Real, E::Vector{Float64}, S::Vector{Float64}) = new(bond, solver, Float64(ϵ), E, S)
    DMRGsiteinfo() = new(BondInfo(), Lanczosinfo(), 0.0, Inf, 0.0)
end

mutable struct CBEinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    err::Float64
    CBEinfo(direction::SweepDirection, ϵ::Real) = new{typeof(direction)}(direction, Float64(ϵ))
    CBEinfo(direction::SweepDirection) = new{typeof(direction)}(direction, 0.0)
end

mutable struct TDVPinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    n::Int64
    err::Float64
    lnZ::Float64
    E::Float64
    S::Vector{Float64}
    TDVPinfo(bond::BondInfo, solver::SolverInfo, n::Int64, ϵ::Real, lnZ::Real, E::Real, S::Vector{Float64}) = new(bond, solver, n, Float64(ϵ), Float64(lnZ), Float64(E), S)
    TDVPinfo(info::TDVPinfo) = new(BondInfo(), Lanczosinfo(), info.n, 0.0, info.lnZ, info.E, info.S)
    TDVPinfo() = new(BondInfo(), Lanczosinfo(), 0, 0.0, 0.0, 0.0, Float64[])
    TDVPinfo(lnZ::Real) = new(BondInfo(), Lanczosinfo(), 0, 0.0, Float64(lnZ), 0.0, Float64[])
end

mutable struct TDVPsweepinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    bond::BondInfo
    solver::SolverInfo
    err::Float64
    E::Float64
    S::Vector{Float64}
    TDVPsweepinfo(direction::SweepDirection, bond::BondInfo, solver::SolverInfo, ϵ::Real, E::Real, S::Vector{Float64}) = new{typeof(direction)}(direction, bond, solver, Float64(ϵ), Float64(E), S)
    TDVPsweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(), 0.0, 0.0, Float64[])
    TDVPsweepinfo(direction::SweepDirection, err::Real) = new{typeof(direction)}(direction, BondInfo(), Lanczosinfo(), Float64(err), 0.0, Float64[])
end

mutable struct TDVPsiteinfo <: AlgorithmInfo
    bond::BondInfo
    solver::SolverInfo
    err::Float64
    E::Float64
    S::Float64
    TDVPsiteinfo(bond::BondInfo, solver::SolverInfo, ϵ::Real, E::Real, S::Real) = new(bond, solver, Float64(ϵ), Float64(E), Float64(S))
    TDVPsiteinfo() = new(BondInfo(), Lanczosinfo(), 0.0, 0.0, 0.0)
end

mutable struct SETTNinfo <: AlgorithmInfo
    bond::BondInfo
    n::Int64
    err::Float64
    lnZ::Float64
    SETTNinfo(bond::BondInfo, n::Int64, ϵ::Real, lnZ::Real) = new(bond, n, Float64(ϵ), Float64(lnZ))
    SETTNinfo(info::SETTNinfo) = new(BondInfo(), info.n, NaN, NaN)
    SETTNinfo() = new(BondInfo(), 0, NaN, NaN)
end

mutable struct SETTNsweepinfo <: AlgorithmInfo
    bond::BondInfo
    err::Float64
    lnZ::Float64
    SETTNsweepinfo(bond::BondInfo, ϵ::Real, lnZ::Real) = new(bond, Float64(ϵ), Float64(lnZ))
    SETTNsweepinfo(err::Real) = new(BondInfo(), Float64(err), 0.0)
    SETTNsweepinfo() = new(BondInfo(), 0.0, 0.0)
end

mutable struct Algebrainfo <: AlgorithmInfo
    bond::BondInfo
    n::Int64
    err::Float64
    truncerr::Float64
    Algebrainfo(bond::BondInfo, n::Int64, ϵ::Real, truncerr::Real = 0) = new(bond, n, Float64(ϵ), Float64(truncerr))
    Algebrainfo(info::Algebrainfo) = new(BondInfo(), info.n, 0.0, 0.0)
    Algebrainfo() = new(BondInfo(), 0, 0.0, 0.0)
end

mutable struct Algebrasweepinfo{Dir} <: AlgorithmInfo where Dir
    direction::SweepDirection
    bond::BondInfo
    err::Float64
    truncerr::Float64
    Algebrasweepinfo(direction::SweepDirection, bond::BondInfo, ϵ::Real, truncerr::Real = 0) = new{typeof(direction)}(direction, bond, Float64(ϵ), Float64(truncerr))
    Algebrasweepinfo(direction::SweepDirection) = new{typeof(direction)}(direction, BondInfo(), 0.0, 0.0)
end

mutable struct Algebrasiteinfo <: AlgorithmInfo
    bond::BondInfo
    err::Float64
    truncerr::Float64
    Algebrasiteinfo(bond::BondInfo, ϵ::Real, truncerr::Real = 0) = new(bond, Float64(ϵ), Float64(truncerr))
    Algebrasiteinfo() = new(BondInfo(), 0.0, 0.0)
end

# function merge(A::DMRGsweepinfo{dir₁},B::DMRGsweepinfo{dir₂}) where {dir₁,dir₂}
#     @assert dir₁ == dir₂ "direction mismatch"
#     return DMRGsweepinfo(sch₁,merge(A.bond, B.bond),merge(A.solver, B.solver),min(A.E,B.E),max(A.σE,B.σE))
# end

function TimerOutputs.merge!(A::DMRGinfo,B::DMRGsweepinfo{dir}) where dir
    merge!(A.bond, B.bond)
    merge!(A.solver, B.solver)
    A.err = max(A.err,B.err)
    # A.E = B.E
    A.E = vcat(A.E,B.E)
    A.S = vcat(A.S,B.S)
    dir <: R2L && (A.n += 1)
    return A
end

function TimerOutputs.merge!(A::TDVPinfo,B::TDVPsweepinfo{dir}) where dir
    merge!(A.bond, B.bond)
    merge!(A.solver, B.solver)
    A.err = B.err
    # A.E = B.E
    A.S = vcat(A.S,B.S)
    return A
end

function TimerOutputs.merge!(A::Algebrainfo,B::Algebrasweepinfo{dir}) where dir
    merge!(A.bond, B.bond)
    A.err = B.err
    A.truncerr = B.truncerr
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
    A.err = max(A.err,B.err)
    push!(A.S,B.S)
    if T₁ <: DMRGsweepinfo && T₂ <: DMRGsiteinfo   
        push!(A.E,B.E)
        # push!(A.S,B.S)
    end

    return A
end

function TimerOutputs.merge!(A::Algebrasweepinfo,B::Algebrasiteinfo)
    merge!(A.bond, B.bond)
    A.err = max(A.err,B.err)
    A.truncerr = max(A.truncerr,B.truncerr)
    return A
end

function TimerOutputs.merge!(info1::DMRGsiteinfo, info2::CBEinfo)
    info1.err = max(info1.err, info2.err)
end

function TimerOutputs.merge!(info1::TDVPsiteinfo, info2::CBEinfo)
    info1.err = info1.err + info2.err
end

function TimerOutputs.merge!(info1::T,info2::BondInfo) where T<:Union{DMRGsiteinfo,TDVPsiteinfo}
    info1.bond = info2
    info1.S = info2.S
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

function update!(A::BondInfo,B::Union{MPSTensor{<:Number, 2},DenseMPOTensor{<:Number, 2}})
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
    println(io,info.bond,", ProjErr = $(info.err), TruncErr = $(info.truncerr)")
end

function Base.show(io::IO,info::SETTNinfo)
    println(io,info.bond,", lnZ = $(info.lnZ), lnZ Err = $(info.err)")
end

function Base.show(io::IO,info::DMRGsweepinfo)
    x = filter(!isnan,info.E)
    y = filter(!isnan,info.S)
    # println(io,info.bond,", K = $(info.solver.numiter), TruncError = $(info.err), E = $(info.E[end]), σE = $(std(x)), ⟨E⟩ = $(sum(x)/length(x))")
    # println(io,info.bond,", σS = $(std(y)),  ⟨E⟩ = $(sum(y)/length(y)), K = $(info.solver.numiter), TruncError = $(info.err), E = $(info.E[end]), σE = $(std(x)), ⟨E⟩ = $(sum(x)/length(x))")
    println(io,info.bond,", σS = $(std(y)), ⟨S⟩ = $(sum(y)/length(y)), max |ΔS| = $(maximum(abs.(diff(y))))")
    println("E = $(info.E[end]), σE = $(std(x)), ⟨E⟩ = $(sum(x)/length(x))")
    println("K = $(info.solver.numiter), TruncError = $(info.err)")
end

function Base.show(io::IO,info::TDVPsweepinfo)
    # println(io,info.bond,", K = $(info.solver.numiter), TruncError = $(info.err)")
    y = filter(!isnan,info.S)
    println(io,info.bond,", σS = $(std(y)), ⟨S⟩ = $(sum(y)/length(y)), max |ΔS| = $(maximum(abs.(diff(y))))")
    println("K = $(info.solver.numiter), TruncError = $(info.err)")
    println("E = $(info.E)")
end

function Base.show(io::IO,info::SETTNsweepinfo)
    println(io,info.bond,", lnZ = $(info.lnZ), AlgebraErr = $(info.err)")
end

function Base.show(io::IO,info::Algebrasweepinfo)
    println(io,info.bond,", ProjErr = $(info.err), TruncErr = $(info.truncerr)")
end

function Base.show(io::IO,info::BondInfo)
    print(io,"D( $(info.Deff) => $(info.D) ), S = $(info.S)")
end


