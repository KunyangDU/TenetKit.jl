using TensorKit,AbstractTrees

"""
    build_mps(d::GradedSpace,L::Int64; tail::Bool = true, perm::Bool = true, kwargs...) -> Vector{::Number, ::Tuple}
Given Local Hilbert Space `d` and system size `L`\\
Return a `Vector{::Number, ::Tuple}` with :
- a probability `p` calculated by final quantum numbr;
- a tuple of MPS/MPO local tensor for MPS/MPO corresponding to `p`, giving samples for stoMPS algorithm;
# Kwargs
    tail::Bool = true
Whethe to incoporate trivial space, giving local tensors for MPS or MPO. (`true` for MPO)

    perm::Bool = true
Whethe to permute the first MPO left auxilliary bond and upper trivial space.

    issplit::Bool = false
Whether to split the D* ≠ 1 representation space in building representation tree, such as Irrep[ℤ₂×SU₂](0,0) sector of ℤ₂×SU₂ fermion. See also `maketree!`.
# Examples
```jldoctest
julia> samples = build_mps(Rep[ℤ₂×SU₂]((0,0) => 2, (1,1//2) => 1), 3);
julia> sum(map(x -> x[1], samples)) ≈ 1
true
julia> map(x -> space(x[2][1]), samples)
13-element Vector{TensorMapSpace{GradedSpace{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, Int64}}, 2, 2}}:
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>8) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>4))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>4))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 1)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 1)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 3/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 1)=>1))

julia> samples = build_mps(Rep[ℤ₂×SU₂]((0,0) => 2, (1,1//2) => 1), 3; perm = false);
julia> sum(map(x -> x[1], samples)) ≈ 1
true
julia> map(x -> space(x[2][1]), samples)
13-element Vector{TensorMapSpace{GradedSpace{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, Int64}}, 2, 2}}:
 (Rep[ℤ₂ × SU₂]((0, 0)=>8) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>4))
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>4))
 (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((1, 1/2)=>2))
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 1)=>1))
 (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 1)=>1))
 (Rep[ℤ₂ × SU₂]((1, 3/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 1)=>1))

julia> samples = build_mps(Rep[ℤ₂×SU₂]((0,0) => 2, (1,1//2) => 1), 3; tail = false);
julia> sum(map(x -> x[1], samples)) ≈ 1
true
julia> map(x -> space(x[2][1]), samples)
13-element Vector{TensorMapSpace{GradedSpace{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, Int64}}, 2, 1}}:
 (Rep[ℤ₂ × SU₂]((0, 0)=>8) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((0, 0)=>4)
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((0, 0)=>4)
 (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
 (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
 (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((0, 0)=>1)
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((0, 0)=>1)
 (Rep[ℤ₂ × SU₂]((0, 0)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>4) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
 (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
 (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((0, 1)=>1)
 (Rep[ℤ₂ × SU₂]((0, 1)=>2) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((0, 1)=>1)
 (Rep[ℤ₂ × SU₂]((1, 3/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← Rep[ℤ₂ × SU₂]((0, 1)=>1)

julia> samples = build_mps(Rep[ℤ₂×SU₂]((0,0) => 2, (1,1//2) => 1), 3; issplit = true);
julia> sum(map(x -> x[1], samples)) ≈ 1
true
julia> map(x -> space(x[2][1]), samples)
151-element Vector{TensorMapSpace{GradedSpace{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{Z2Irrep, SU2Irrep}}, Int64}}, 2, 2}}:
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((1, 1/2)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>2, (1, 1/2)=>1)) ← (Rep[ℤ₂ × SU₂]((0, 0)=>1) ⊗ Rep[ℤ₂ × SU₂]((0, 0)=>1))
 ⋮
```
"""
build_mps(d::GradedSpace,L::Int64; tail::Bool = true, perm::Bool = true, kwargs...) = build_mps(d, getconfig(d,L;kwargs...); tail = tail, perm = perm)

"""
    getconfig(d::GradedSpace, L::Int64; kwargs...) -> Vector{Tuple{Float64,Tuple}}
Given Local Hilbert Space `d` and system size `L`\\
Return a `Vector{Tuple{Float64,Tuple}}` with :
- a probability `p` calculated by final quantum numbr;
- a tuple of representation space corresponding to `p`, giving auxilliary space of a MPS for stoMPS algorithm;
- a tuple of projector for split sampling of non-single D*
# Examples
```jldoctest
julia> getconfig(Rep[SU₂](1//2 => 1),2)
2-element Vector{Tuple{Float64, Tuple}}:
 (0.5, (Rep[SU₂](0=>1), Rep[SU₂](1/2=>1), Rep[SU₂](0=>1)))
 (0.5, (Rep[SU₂](1=>1), Rep[SU₂](1/2=>1), Rep[SU₂](0=>1)))

julia> getconfig(Rep[U₁×SU₂]((-1,0) => 1, (0,1//2) => 1, (1,0) => 1), 2)
10-element Vector{Any}:
 (0.1, (Rep[U₁ × SU₂]((0, 0)=>1), Rep[U₁ × SU₂]((1, 0)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((2, 0)=>1), Rep[U₁ × SU₂]((1, 0)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((1, 1/2)=>1), Rep[U₁ × SU₂]((1, 0)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((0, 0)=>1), Rep[U₁ × SU₂]((-1, 0)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((-2, 0)=>1), Rep[U₁ × SU₂]((-1, 0)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((-1, 1/2)=>1), Rep[U₁ × SU₂]((-1, 0)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((0, 0)=>1), Rep[U₁ × SU₂]((0, 1/2)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((1, 1/2)=>1), Rep[U₁ × SU₂]((0, 1/2)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((-1, 1/2)=>1), Rep[U₁ × SU₂]((0, 1/2)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
 (0.1, (Rep[U₁ × SU₂]((0, 1)=>1), Rep[U₁ × SU₂]((0, 1/2)=>1), Rep[U₁ × SU₂]((0, 0)=>1)), (nothing, nothing, nothing))
```
"""
getconfig(d::GradedSpace, L::Int64;kwargs...) = getconfig(buildtree!(d,L;kwargs...))

"""
    buildtree!(d::GradedSpace, i::Int64;kwargs...) -> RepresentationTreeNode
Given local Hilbert space `d` and system size `i`, 
return the same `RepresentationTreeNode` with filled children by recursively span onsite space, using `fuse(d,...)`.
# Kwargs
    A::GradedSpace = trivial(d)
Initial representation space of MPS, which is usually trivial space of local Hilbert space, i.e., `trivial(d)`.

    issplit::Bool = false
Whether to split the D* ≠ 1 representation space, such as Irrep[ℤ₂×SU₂](0,0) sector of ℤ₂×SU₂ fermion.

    projector::Union{Nothing, AbstractTensorMap} = nothing
The projector for different subspace of the D* ≠ 1 representation space.
# Examples
```jldoctest
julia> buildtree!(RepresentationTreeNode(Rep[SU₂](1//2 => 1)),2)
Rep[SU₂](0=>1)
└─ Rep[SU₂](1/2=>1)
   ├─ Rep[SU₂](0=>1)
   └─ Rep[SU₂](1=>1)

julia> buildtree!(RepresentationTreeNode(Rep[U₁×SU₂]((-1,0) => 1, (0,1//2) => 1, (1,0) => 1)),2)
Rep[U₁ × SU₂]((0, 0)=>1)
├─ Rep[U₁ × SU₂]((1, 0)=>1)
│  ├─ Rep[U₁ × SU₂]((0, 0)=>1)
│  ├─ Rep[U₁ × SU₂]((2, 0)=>1)
│  └─ Rep[U₁ × SU₂]((1, 1/2)=>1)
├─ Rep[U₁ × SU₂]((-1, 0)=>1)
│  ├─ Rep[U₁ × SU₂]((0, 0)=>1)
│  ├─ Rep[U₁ × SU₂]((-2, 0)=>1)
│  └─ Rep[U₁ × SU₂]((-1, 1/2)=>1)
└─ Rep[U₁ × SU₂]((0, 1/2)=>1)
   ├─ Rep[U₁ × SU₂]((0, 0)=>1)
   ├─ Rep[U₁ × SU₂]((1, 1/2)=>1)
   ├─ Rep[U₁ × SU₂]((-1, 1/2)=>1)
   └─ Rep[U₁ × SU₂]((0, 1)=>1)

julia> buildtree!(RepresentationTreeNode(Rep[ℤ₂×SU₂]((0,0) => 2,(1,1//2)=>1)),2)
Rep[ℤ₂ × SU₂]((0, 0)=>1)
├─ Rep[ℤ₂ × SU₂]((0, 0)=>2)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  └─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
└─ Rep[ℤ₂ × SU₂]((1, 1/2)=>1)
   ├─ Rep[ℤ₂ × SU₂]((0, 0)=>1)
   ├─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
   └─ Rep[ℤ₂ × SU₂]((0, 1)=>1)

julia> buildtree!(Rep[ℤ₂×SU₂]((0,0) => 2,(1,1//2)=>1),2; issplit = true)
Rep[ℤ₂ × SU₂]((0, 0)=>1)
├─ Rep[ℤ₂ × SU₂]((0, 0)=>2)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
│  └─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
├─ Rep[ℤ₂ × SU₂]((0, 0)=>2)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((0, 0)=>4)
│  ├─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
│  └─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
└─ Rep[ℤ₂ × SU₂]((1, 1/2)=>1)
   ├─ Rep[ℤ₂ × SU₂]((0, 0)=>1)
   ├─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
   ├─ Rep[ℤ₂ × SU₂]((1, 1/2)=>2)
   └─ Rep[ℤ₂ × SU₂]((0, 1)=>1)
```
"""
function buildtree!(d::GradedSpace, i::Int64;kwargs...)
    A = get(kwargs,:A,trivial(d))
    issplit = get(kwargs,:issplit, false)
    projector = get(kwargs, :projector, nothing)
    return buildtree!(RepresentationTreeNode{GradedSpace}(A, d, issplit, projector), i)
end

abstract type AbstractTreeNode end

mutable struct RepresentationTreeNode{T} <: AbstractTreeNode where T <: GradedSpace
    A::Union{Nothing,T}
    d::Union{Nothing,T}
    issplit::Bool
    projector::Union{Nothing, AbstractTensorMap}
    parent::Union{Nothing,RepresentationTreeNode}
    children::Vector{RepresentationTreeNode}

    function RepresentationTreeNode(
        A::GradedSpace,d::GradedSpace,
        issplit::Bool,
        projector::Union{Nothing, AbstractTensorMap},
        parent::RepresentationTreeNode,
        children::Vector{RepresentationTreeNode}=RepresentationTreeNode[],
    )
        return new{GradedSpace}(A,d,issplit,projector,parent,children)
    end

    function RepresentationTreeNode{GradedSpace}(
        A::GradedSpace,d::GradedSpace,
        issplit::Bool,
        projector::Union{Nothing, AbstractTensorMap},
        children::Vector{RepresentationTreeNode}=RepresentationTreeNode[],
    )
        return new{GradedSpace}(A,d,issplit,projector,nothing,children)
    end

    function RepresentationTreeNode(
        d::GradedSpace,children::Vector{RepresentationTreeNode}=RepresentationTreeNode[],
    )
        return new{GradedSpace}(trivial(d), d, false, isometry(trivial(d),trivial(d)), nothing, children)
    end
end

AbstractTrees.nodevalue(node::RepresentationTreeNode) = node.A
AbstractTrees.parent(node::AbstractTreeNode) = node.parent
AbstractTrees.children(node::AbstractTreeNode) = node.children
AbstractTrees.ParentLinks(::Type{AbstractTreeNode}) = StoredParents()
AbstractTrees.ChildIndexing(::Type{AbstractTreeNode}) = IndexedChildren()
AbstractTrees.NodeType(::Type{AbstractTreeNode}) = HasNodeType()
AbstractTrees.nodetype(::T) where T <: AbstractTreeNode = T

#= tools =#

TensorKit.space(A::T,d::Int64) where T <: Union{ProductSector, AbstractIrrep} = GradedSpace{typeof(A)}(A => d)
Main.eltype(node::AbstractTreeNode) = typeof(node.A)
trivial(::GradedSpace{I, D}) where {I, D} = GradedSpace{I,D}(TensorKit.SortedVectorDict(one(I) => 1), false)
TensorKit.axpy!(α::Number,x::AbstractTensorMap,::Nothing) = rmul!(x,α)

#= utils =#

function addchild!(node::AbstractTreeNode, child::AbstractTreeNode)
    isnothing(child.parent) ? child.parent = node : @assert child.parent == node
    push!(node.children, child)
    return nothing
end

addchild!(node::T, A::GradedSpace, d::GradedSpace, issplit::Bool, projector::Union{Nothing,AbstractTensorMap}) where T <: RepresentationTreeNode = addchild!(node,T(A,d,issplit,projector))

function ancestors(n::AbstractTreeNode) 
    acs = Vector{typeof(n)}()
    while !isnothing(n)
        push!(acs,n)
        n = n.parent
    end
    return acs
end

function Base.show(io::IO,Root::AbstractTreeNode)
    print_tree(Root;maxdepth = 16)
    return nothing
end

function buildtree!(n::RepresentationTreeNode,i::Int64)
    n.issplit && isnothing(n.projector) && (n.projector = isometry(n.A,n.A))
    i == 0 && return n
    for s in sectors(n.A)
        fspace = fuse(space(s,dim(n.A,s)),n.d')
        for ss in sectors(fspace)
            if n.issplit
                for i in 1:dim(fspace,ss)
                    projector = TensorMap(zeros,space(ss,1),space(ss,dim(fspace,ss))) 
                    block(projector,ss)[1,i] = 1
                    addchild!(n,space(ss,dim(fspace,ss)),n.d,n.issplit,projector)
                end
            else
                addchild!(n,space(ss,dim(fspace,ss)),n.d,n.issplit,nothing)
            end
        end
    end
    # build tree recursively
    for c in n.children
        buildtree!(c,i-1)
    end
    return n
end

function getconfig(n0::RepresentationTreeNode)
    lvs = collect(Leaves(n0))
    cfg = []
    N = sum(map(x -> sum(map(y -> dim(x.A,y),sectors(x.A))), lvs))
    for l in lvs
        acs = ancestors(l)
        push!(cfg,(1/length(lvs), Tuple(map(x -> x.A,acs)), Tuple(map(x -> x.projector, acs))))
    end
    return cfg
end

function _global_identity_check(
        ds::Vector = [
        Rep[U₁](1//2 => 1, -1//2 => 1),
        Rep[SU₂](1//2 => 1), 
        Rep[U₁×SU₂]((0,1//2) => 1,(-1,0) => 1,(1,0) => 1), 
        Rep[U₁×U₁]((-1, 0) => 1, (1,0) => 1, (0, -1 // 2) => 1, (0, 1 // 2) => 1),
        Rep[ℤ₂×SU₂]((0,0) => 2,(1,1//2)=>1),
        Rep[U₁×SU₂]((0,1//2) => 1, (-1,0) => 1),
        Rep[U₁×U₁]((-1, 0) => 1, (0, -1 // 2) => 1, (0, 1 // 2) => 1)
    ],
    L::Int64 = 3;
    issplit::Bool = false
    )

    for d in ds

        local A = nothing
        local config = getconfig(d,L;issplit=issplit)

        for (p,aux,projector) in config
            local x = nothing
            for i in 1:L
                tmp = permute(isometry(d ⊗ aux[i+1]', aux[i]'),(3,1),(2,))
                if issplit
                    @tensor tmp[-1,-2;-3] ≔ projector[i][-1,1] * tmp[1,-2,2] * projector[i+1]'[2,-3]
                end
                if i == 1
                    x = tmp
                else
                    @tensor tmp′[-1,-2,-3;-4] ≔ x[-1,-2,1] * tmp[1,-3,-4]
                    iso = isometry(fuse(space(tmp′)[2],space(tmp′)[3]), space(tmp′)[2] ⊗ space(tmp′)[3])
                    @tensor x′[-1,-2;-3] ≔ iso[-2,1,2] * tmp′[-1,1,2,-3]
                    x = x′
                end
            end
            @tensor I′[-1;-2] ≔ x[1,-1,2] * x'[2,1,-2]
            A = axpy!(p,I′,A)
        end

        rmul!(A,length(config))
        @assert A ≈ isometry(space(A)) "∑ᵢpᵢ|ψᵢ⟩⟨ψᵢ| ≠ 𝔼 at d = $d"

        printstyled("✔ Global Check PASS: "; color=:green, bold=true)
        println(" ∑ᵢpᵢ|ψᵢ⟩⟨ψᵢ| = 𝔼 with d = $(d), issplit = $(issplit)")
    end

    return nothing
end

function _local_identity_check(
    ds::Vector = [
        Rep[U₁](1//2 => 1, -1//2 => 1),
        Rep[SU₂](1//2 => 1), 
        Rep[U₁×SU₂]((0,1//2) => 1,(-1,0) => 1,(1,0) => 1), 
        Rep[U₁×U₁]((-1, 0) => 1, (1,0) => 1, (0, -1 // 2) => 1, (0, 1 // 2) => 1),
        Rep[ℤ₂×SU₂]((0,0) => 2,(1,1//2)=>1),
        Rep[U₁×SU₂]((0,1//2) => 1, (-1,0) => 1),
        Rep[U₁×U₁]((-1, 0) => 1, (0, -1 // 2) => 1, (0, 1 // 2) => 1)
    ],
    L::Int64 = 3;
    issplit::Bool = false
    )

    for d in ds
        
        local n0 = buildtree!(d,L;issplit = issplit)
        local parents = [n0,]
        local A₀ = isometry(n0.d, n0.d)

        for _ in 1:treeheight(n0)

            local A = nothing
            local N = issplit ? sum(map(x -> sum(map(y -> dim(y),sectors(x.A))),parents)) : sum(map(x -> dim(x.A), parents))
            for p in parents, c in p.children
                t = permute(isometry(p.d ⊗ p.A', c.A'),(3,1),(2,))
                if n0.issplit
                    @tensor t[-1,-2;-3] ≔ c.projector[-1,1] * t[1,-2,2] * p.projector'[2,-3]
                end
                @tensor I′[-1;-2] ≔ t[1,-1,2] * t'[2,1,-2]
                A = axpy!(1/N, I′, A)
            end
            @assert A ≈ A₀ "∑ᵢpᵢ|ψᵢ⟩⟨ψᵢ| ≠ 𝔼 at d = $d"
            
            parents = vcat(map(x -> x.children, parents)...)
        end
        printstyled("✔  Local Check PASS: "; color=:green, bold=true)
        println(" ∑ᵢpᵢ|ψᵢ⟩⟨ψᵢ| = 𝔼 with d = $(d), issplit = $(issplit)")
    end

    return nothing
end


function _build_mps_tensor(spl::GradedSpace, d::GradedSpace, spr::GradedSpace, ::Nothing, ::Nothing;
    tail::Bool = true, perm::Bool = false)
    if tail 
        if perm
            return permute(isometry(spr ⊗ d' , trivial(d) ⊗ fuse(spl',trivial(d))')',(1,4),(2,3))*1
        else
            return permute(isometry(d ⊗ spr', trivial(d) ⊗ spl'),(4,1),(3,2))
        end
    else
        return permute(isometry(d ⊗ spr', spl'),(3,1),(2,))
    end 
end

function _build_mps_tensor(spl::GradedSpace, d::GradedSpace, spr::GradedSpace, pl::AbstractTensorMap, pr::AbstractTensorMap;
    tail::Bool = true, perm::Bool = false)
    x = _build_mps_tensor(spl,d,spr,nothing,nothing;tail = tail, perm = perm)
    if tail 
        if perm 
            pl = permute(pl,(2,),(1,))
            isol = isometry(fuse(space(pl)[1],trivial(d)),space(pl)[1])
            isor = isometry(space(pl)[2]', fuse(space(pl)[2]',trivial(d)))
            @tensor pl[-1;-2] ≔ isol[-1,1] * pl[1,2] * isor[2,-2]
            return @tensor x[-1,-2;-3,-4] ≔ x[-1,-2,1,2] * pl[1,-3] * pr'[2,-4]
        else
            return @tensor x[-1,-2;-3,-4] ≔ pl[-1,1] * x[1,-2,-3,2] * pr'[2,-4]
        end
    else
        return @tensor x[-1,-2;-3] ≔ pl[-1,1] * x[1,-2,2] * pr'[2,-3]
    end
end

function build_mps(d::GradedSpace, p::Number,aux::Tuple,proj::Tuple; tail::Bool = true, perm::Bool = true)
    mps = []
    for i in 1:length(aux)-1
        push!(mps,_build_mps_tensor(aux[i],d,aux[i+1],proj[i:i+1]...;tail = tail, perm = i == 1 ? perm : false))
    end
    return (p,mps)
end

function build_mps(d::GradedSpace, config::Vector; tail::Bool = true, perm::Bool = true)
    totalmps = []
    for c in config
        push!(totalmps,build_mps(d, c...;tail = tail, perm = perm))
    end
    return totalmps
end

_local_identity_check()
_local_identity_check(issplit = true)
_global_identity_check()
_global_identity_check(issplit = true)

L = 3

# d = Rep[ℤ₂×SU₂]((0,0) => 2, (1,1//2) => 1)
# d = Rep[U₁×SU₂]((0,1//2) => 1, (-1,0) => 1, (1,0) => 1)
# d = Rep[SU₂](1//2 => 1)
# d = Rep[U₁](-1//2 => 1, 1//2 => 1)
# d = Rep[U₁×U₁]((-1,0) => 1, (1,0) => 1, (0,1//2) => 1, (0,-1//2) => 1)

d = Rep[U₁×SU₂]((0,1//2) => 1, (-1,0) => 1)
# d = Rep[U₁×U₁]((-1,0) => 1, (0,1//2) => 1, (0,-1//2) => 1)

samples = build_mps(d, L)
# samples = build_mps(d, L; tail = false)
# samples = build_mps(d, L; perm = false)
# samples = build_mps(d, L; issplit = true)
# samples = build_mps(d, L; tail = false,issplit = true)
# samples = build_mps(d, L; perm = false,issplit = true)

@assert sum(map(x -> x[1],samples)) ≈ 1
# map(x -> (x[2][1]),samples)

samples[3]

