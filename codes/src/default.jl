
global DMRGDefaultLanczos = KrylovKit.Lanczos(;
     krylovdim = 8,
     maxiter = 10,
     tol = 1e-8,
     orth = ModifiedGramSchmidt(),
     eager = true,
     verbosity = 0
)

global TDVPDefaultLanczos = KrylovKit.Lanczos(;
     krylovdim = 32,
     maxiter = 1,
     tol = 1e-8,
     orth = ModifiedGramSchmidt(),
     eager = true,
     verbosity = 0
)

