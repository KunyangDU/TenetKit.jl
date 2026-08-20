using Integrals
f(x, p) = sin(x * p)
p = 1.7
domain = (-2, 5) # (lb, ub)
prob = IntegralProblem(f, domain, p)
sol = solve(prob, QuadGKJL())