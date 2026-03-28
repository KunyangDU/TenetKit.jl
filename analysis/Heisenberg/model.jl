

function window(x::Number)
    if 1/2 < x ≤ 1
        return -2*(x-1)^3
    elseif 0 ≤ x ≤ 1/2
        return 1 - 6x^2 + 6x^3
    elseif -1/2 ≤ x < 0
        return 1 - 6x^2 - 6x^3
    elseif -1 ≤ x < -1/2
        return 2*(x+1)^3
    else
        return 0
    end
end

