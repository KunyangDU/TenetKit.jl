struct SingleSite <: SweepScheme SingleSite() = new() end
struct DoubleSite <: SweepScheme DoubleSite() = new() end

# struct singlesite <: SingleSite singlesite() = new() end
# struct doublesite <: DoubleSite doublesite() = new() end
struct randSVD <: CBEscheme 
    λ::Float64
    randSVD(λ::Float64) = new(λ)
    randSVD() = new(NaN)
end
struct fullSVD <: CBEscheme
    # N::Int64
    # fullSVD(N::Int64) = new(N)
    fullSVD() = new()
end
struct dynamicSVD <: CBEscheme
    λ::Float64
    N::Int64
    dynamicSVD() = new(NaN,NaN)
    dynamicSVD(λ::Float64,N::Int64) = new(λ,N)
end

# struct randSVD <: CBEscheme 
#     complexity::Function
#     function randSVD()
#         F(D,d,w) = D^2 * d * w
#         return new(F) 
#     end
# end
