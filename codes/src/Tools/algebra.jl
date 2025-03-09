


function ApproxReal(Qi::Number;tol::Float64=1e-1)
    imag(Qi) <= tol && return real(Qi)
    @error "$(Qi) not real"
end

function centralize(data::Union{Vector,OrdinalRange,StepRangeLen})
    return [(data[i] + data[i+1]) / 2 for i in 1:(length(data) - 1)]
end

function showdomain(M::AbstractTensorMap)
    @show codomain(M),domain(M)
end

function showQuantSweep(lsQ::Vector;name::String = "Quantity")
    for (iq,q) in enumerate(lsQ)
        println("$name\t$iq\t$q")
    end
end

