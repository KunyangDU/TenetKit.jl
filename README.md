# iMPS.jl

Personal code for finite MPS simulations, **which is still under great development**.

## Features

### Versatility

Now this code contains:

1. Density matrix renormalization group ([DMRG](https://en.wikipedia.org/wiki/Density_matrix_renormalization_group)) for calculating groundstates.
2. Time-dependent variational principle ([TDVP](https://link.aps.org/doi/10.1103/PhysRevB.94.165116)) for real and imaginary time evolution, which serves for dynamics (i.e., spectrum function and structure factor) and finite temperature simulations (i.e., tangent space tensor renormalization group, [tanTRG](https://link.aps.org/doi/10.1103/PhysRevLett.130.226502)), respectively.
3. Series-expansion thermal tensor network ([SETTN](https://link.aps.org/doi/10.1103/PhysRevB.95.161104)) for finite temperature simulations.
4. Controlled Bond Expansion ([CBE](https://doi.org/10.1103/PhysRevLett.130.246402)): algorithm for single site DMRG/TDVP to change local quantum numebr self-consistently, which effectively reduce the time and memory consumed.

### Examples

This code has been applied to many models in 1D and 2D finite lattice (square lattice mostly), including:

* Hubbard model: compared with [ED](https://github.com/KunyangDU/iED.jl.git)

![Fermi Hubbard](slides/mdfig/hubbard_U=0.png "Fermi Hubbard, U=0")
![Fermi Hubbard](slides/mdfig/hubbard_U=8.png "Fermi Hubbard, U=8")

calculated with $U=0$ (free fermion):

![Free fermion](slides/mdfig/free%20fermion.png "free fermion")

The tutorial is to be added.

### Announcement

Some algorithms in this code is developed by [ CQM2 group](https://www.cqm2itp.com/) which works on purification-based finite-temperature simulations. Relevant packages are listed below:

* [FiniteMPS.jl](https://github.com/Qiaoyi-Li/FiniteMPS.jl)

## TODO

* Some local tensor can be redefined to improve the memory performance.

## Acknowledgments

The following packages has been used:

* [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) for basic tensor operations.
* [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) for nested multi-threaded BLAS.
