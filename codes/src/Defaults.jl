
global DMRGDefaultLanczos = Krylovalgo(KrylovKit.Lanczos(;
     krylovdim = 8,
     maxiter = 1,
     tol = 1e-6,
     orth = ModifiedGramSchmidt(),
     eager = true,
     verbosity = 0
))
 
global TDVPDefaultLanczos = Krylovalgo(KrylovKit.Lanczos(;
     krylovdim = 32,
     maxiter = 1,
     tol = 1e-8,
     orth = ModifiedGramSchmidt(),
     eager = true,
     verbosity = 0
))

global TDVPDefaultChebyshev = Chebyshev(tol=1e-10, maxiter=500)

global HamiltonianBoundDefaultLanczos = Krylovalgo(KrylovKit.Lanczos(krylovdim=16, maxiter=2, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))

# global TDVPDefaultGMRES = Krylovalgo(KrylovKit.GMRES(;
#      krylovdim = 16,
#      maxiter = 2,
#      tol = 1e-8,
#      orth = ModifiedGramSchmidt(),
#      verbosity = 0
# ))
