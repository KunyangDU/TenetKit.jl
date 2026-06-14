# Efficient higher-order matrix product operators for time evolution

# Maarten Van Damme1, Jutho Haegeman1, Ian McCulloch2,3,4 and Laurens Vanderstraeten1,5⋆

1 Department of Physics and Astronomy, University of Ghent, Belgium 

2 School of Mathematics and Physics, The University of Queensland, Australia 

3 Frontier Center for Theory and Computation, National Tsing Hua University, Hsinchu 30013, Taiwan 

4 Department of Physics, National Tsing Hua University, Hsinchu 30013, Taiwan 5 Center for Nonlinear Phenomena and Complex Systems, Université Libre de Bruxelles, Belgium 

⋆ laurens.vanderstraeten@ulb.be 

# Abstract

We introduce a systematic construction of higher-order matrix product operator (MPO) approximations of the time evolution operator for generic (short and long range) onedimensional Hamiltonians. We demonstrate the utility of our construction, by showing an order of magnitude speedup in simulation cost compared to conventional first-order MPO time evolution schemes. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/9f7d29c7848403b33f28f779d8eb951d3d884d9713b70efac81c473ead5eb5e2.jpg)


Copyright M. Van Damme et al. 

This work is licensed under the Creative Commons 

Attribution 4.0 International License. 

Published by the SciPost Foundation. 

Received 17-03-2023 

Accepted 07-10-2024 

Published 15-11-2024 

doi:10.21468/SciPostPhys.17.5.135 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/04e17fc5c146013564d1deb28b765784360070cbc7894c0772acd730a83c9ae6.jpg)


Check for updates 

# Contents

1 Introduction 2 

2 Matrix product states and matrix product operators 3 

2.1 General notation 3 

2.2 Applying an MPO to an MPS 4 

2.3 MPO representation of extensive Hamiltonians 4 

2.4 Examples 5 

2.5 Powers of MPOs 7 

3 From powers of the Hamiltonian to extensive MPOs 8 

4 Exact compression steps 11 

5 Incorporating higher-order terms 12 

6 Approximate compressions 14 

Numerical compression 15 

# 8 Benchmarks 15

8.1 Precision of nth order MPO 15 

8.2 Efficiency 16 

8.3 Splitting schemes 17 

8.4 Finite temperature 17 

# 9 Conclusion and outlook 18

# A Explicit expressions 19

# References 19

# 1 Introduction

Some years following the discovery of the density matrix renormalization group (DMRG) [1] algorithm, it was reformulated as a variational method in the language of matrix product states (MPS). This proved to be a fruitful endeavor, as it not only explained the astounding accuracy of DMRG in approximating ground state properties of strongly interacting one-dimensional quantum systems, but it also opened the door to a zoo of algorithms that greatly extend the range of applicability beyond mere ground state properties [2]. 

In particular, it was realized that MPS can also be used to simulate the time evolution of an interacting system. Although the entanglement in a state generically increases under unitary time evolution and the MPS bond dimension would have to grow exponentially, in practice MPS simulations can reach surprisingly long times with high accuracy. Initial algorithms were limited to short-range interacting systems by using the Trotter-Suzuki decomposition of the time-evolution operator [3–5]. This restriction has by now been lifted using more involved algorithms [6–8], allowing one to target even quasi two-dimensional and long-range interacting systems. Still, these methods all rely on evolving states by taking small time steps, to the effect that some non-equilibrium properties remain difficult to calculate up to the desired precision without investing a tremendous amount of CPU or GPU hours. Recently, a new approach [9] based on cluster expansions was introduced to find tensor network approximations of the time evolution operator that are accurate for much larger time steps, but again this approach is limited to short-range interactions. 

In this work, we introduce an approach based on matrix product operators (MPO) [10] that allows us to approximate the full time-evolution operator up to arbitrary order, even for long-range interactions. Our construction can be seen as a higher-order generalization of the $W _ { I } / W _ { I I }$ operators of Ref. [7] or as an extension of the cluster-expansion approach of Ref. [9] to generic Hamiltonians; the form of the MPO reduces to the one of Ref. [7] when considering the first-order case. We demonstrate the utility of such a higher-order scheme in practice, as it is shown to drastically outperform state-of-the-art algorithms for simulating time evolution with MPS. 

The resulting algorithm is both simple to implement and highly flexible, applicable to both finite and infinite systems with arbitrary unit cells and non-abelian symmetries, as long as the Hamiltonian can be represented as an MPO [11]. We provide an example implementation and include analytical MPO expressions that can be implemented and combined with pre-existing tensor network toolboxes. 

# 2 Matrix product states and matrix product operators

In this first section we recapitulate all the essentials on MPS and MPO representations, in order to fix notation and set the stage for the next sections. 

# 2.1 General notation

A matrix product state (MPS) is represented as 

$$
\left| \Psi \right\rangle_ {\text {finite}} = \boxed {M _ {1}} - \boxed {M _ {2}} - \boxed {M _ {3}} - \boxed {M _ {4}} - \boxed {M _ {5}}, \tag {1}
$$

where the variational parameters are contained within the local complex-valued three-leg tensors M i. The dimensions of the virtual bonds of the MPS tensors are called the bond dimension. Similarly as for states, a matrix product operator (MPO) can be constructed as the contraction of local four-leg tensors 

$$
O _ {\text { finite }} = \boxed {O _ {1}} - \boxed {O _ {2}} - \boxed {O _ {3}} - \boxed {O _ {4}} - \boxed {O _ {5}}. \tag {2}
$$

This representation of a quantum state is size-extensive, in the sense that the state is built up from local objects. The construction can therefore be extended to an infinite system, where the state is built up as an infinite repetition of an n-site unit cell of tensors M i: 

$$
\left| \Psi \right\rangle_ {\text { infinite }} = \dots - \boxed {M _ {1}} - \boxed {\dots} - \boxed {M _ {n}} - \boxed {M _ {1}} - \boxed {\dots} - \boxed {M _ {n}} - \dots . \tag {3}
$$

The norm of such an infinite-system state is given by 

$$
\dots \begin{array}{c} \boxed {M _ {1}} \longrightarrow \boxed {\dots} \longrightarrow \boxed {M _ {n}} \longrightarrow \boxed {M _ {1}} \longrightarrow \boxed {\dots} \longrightarrow \boxed {M _ {n}} \\ \boxed {\bar {M} _ {1}} \longrightarrow \boxed {\dots} \longrightarrow \boxed {\bar {M} _ {n}} \longrightarrow \boxed {\bar {M} _ {1}} \longrightarrow \boxed {\dots} \longrightarrow \boxed {\bar {M} _ {n}} \end{array} , \tag {4}
$$

and is well-defined if the unit-cell transfer matrix, 

$$
\begin{array}{c} \framebox {- - } M _ {1} \framebox {- - } \dots \framebox {- - } M _ {n} \\ \framebox {- - } \bar {M} _ {1} \framebox {- - } \dots \framebox {- - } \bar {M} _ {n} \end{array} \tag {5}
$$

has a unique leading eigenvalue – this is called an injective MPS. In that case the leading eigenvalue is necessarily real-positive, and we can naturally normalize by rescaling the MPS tensors such that the leading eigenvalue of the transfer matrix is set to one.1 

An MPO can be similarly considered directly in the thermodynamic limit, and the expectation value of this MPO with respect to an MPS is characterized by the leading eigenvalue of the triple-layer transfer matrix 

$$
\lambda = \rho_ {\max} \left( \begin{array}{c c c c c} - & M _ {1} & \dots & - & M _ {n} \\ & - & & - & \\ - & O _ {1} & \dots & - & O _ {n} \\ & - & & - & \\ - & M _ {1} & \dots & - & M _ {n} \end{array} \right), \tag {6}
$$

such that we can evaluate 

$$
\lambda = \lim _ {N \rightarrow \infty} \frac {1}{N} \log (\langle \Psi | O | \Psi \rangle) \tag {7}
$$

(where N denotes the diverging system size). If this triple-layer transfer matrix is diagonalizable and has a unique leading eigenvalue, the MPO is called a zero-degree MPO.2 


Table 1: Different methods for applying an MPO to an MPS.


<table><tr><td>algorithm</td><td>scaling</td><td>finite</td><td>infinite</td><td>iterative</td></tr><tr><td>naive [2]</td><td><eq>O(D^{3}\chi^{3})</eq></td><td>√</td><td>√</td><td></td></tr><tr><td>zip-up [6,17]</td><td><eq>O(D\chi^{3})</eq></td><td>√</td><td></td><td></td></tr><tr><td>density matrix algorithm [18]</td><td><eq>O(D^{2}\chi^{3})</eq></td><td>√</td><td>√</td><td></td></tr><tr><td>(i)DMRG [13,14]</td><td><eq>O(D\chi^{3})</eq></td><td>√</td><td>√</td><td>√</td></tr><tr><td>non-linear optimization</td><td><eq>O(D\chi^{3})</eq></td><td>√</td><td>√</td><td>√</td></tr><tr><td>variational uniform MPS [16]</td><td><eq>O(D\chi^{3})</eq></td><td></td><td>√</td><td>√</td></tr></table>

# 2.2 Applying an MPO to an MPS

One of the most basic steps in MPS-based algorithms is the application of an MPO to an MPS. The bond dimension of the resulting MPS is the product of the original MPS and MPO bond dimensions, which becomes intractable after doing a few consecutive MPO applications. Therefore, we want to approximate the result again as an MPS with a smaller bond dimension: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/87f877d04021fc3f8f77cd248cae103edfb18290b2afda6b5a77e8344f716177.jpg)


A natural way to find the tensors $M _ { i } ^ { \prime }$ is to naively apply the MPO of bond dimension D to the MPS of bond dimension $\chi _ { : }$ , yielding an MPS with bond dimension $\chi ^ { \prime } = \chi D$ . In a second step we can then truncate this bond dimension down using the Schmidt decomposition, giving an algorithm scaling as $O ( D ^ { 3 } \chi ^ { 3 } )$ ). 

There are more performant schemes available, for example by directly minimizing the 2- norm difference between the left and right hand side of Eq. 8. For finite systems this can be done by a sweeping-like DMRG scheme [13, 14] or with a global non-linear optimization scheme [15] – the latter can be extended to infinite systems by using variational schemes over uniform MPS [12, 16]. Alternatively, for finite systems there is the zip-up method [6, 17] that performs singular-value decompositions without first bringing the state into canonical form. This softens the computational cost considerably, and only leads to small errors. Finally, there is a method based on consecutive truncations of the reduced density matrix [18], also yielding a smaller computational costs. In Table 1 we summarize these different methods with their scope and computational costs. The benchmarks in Sec. 8 were always performed using variational schemes. 

# 2.3 MPO representation of extensive Hamiltonians

A generic spin-chain Hamiltonian H can be represented as an MPO, with the local MPO tensor having the following substructure [2, 19, 20]: 

$$
H \sim \left( \begin{array}{c c c} \mathbb {I} & C & D \\ & A & B \\ & & \mathbb {I} \end{array} \right). \tag {9}
$$

The blocks A, B, C and D are all four-leg tensors and I is the identity operator acting on the local Hilbert space: 

$$
\mathbb {I} = \dots , \quad A = \begin{array}{c c} \chi & \\ \hline \end{array} \text {   -   } \begin{array}{c c} \chi & \\ \hline \end{array} , \quad C = \begin{array}{c c} \chi & \\ \hline \end{array} , \tag {10}
$$

$$
B = \stackrel {\chi} {- - } \boxed {1}, \qquad D = \stackrel {\cdot} {- - } \boxed {1} \dots .
$$

The dimensions of the first and last virtual levels is always one (denoted by the dashed line above), but the dimension of the middle level can be larger; this dimension is henceforth called the MPO’s bond dimension χ. We always require that the spectral radius3 of the middle block A is smaller than one. 

This operator is a first-degree MPO [11], in the sense that the expectation value with respect to an injective MPS scales linearly with system size – as it should for a local Hamiltonian. This is reflected in the structure of the triple-layer transfer matrix [Eq. 6], which has a unique dominant eigenvalue with value 1 (provided the MPS is properly normalized), to which is associated a two-dimensional generalized eigenspace, or thus, a two-dimensional Jordan block. Upon taking the N th power, this gives rise to terms scaling as $1 ^ { N }$ , thus constant, as well as terms scaling as $N 1 ^ { N }$ , or thus linearly in N . The prefactor of this last term corresponds exactly to the bulk energy density. 

A particularly insightful way of representing a first-degree MPO is by a finite-state machine [21]: 

$$
\begin{array}{c c c} & D \\ 1 & C & A \\ 2 & B \end{array} \tag {11}
$$

which makes the meaning of the different blocks immediately clear: When going from left to right through the MPO, the virtual level ‘1’ denotes that the Hamiltonian has not yet acted, the virtual level ‘2’ denotes that the Hamiltonian is acting non-trivially and the virtual level ‘3’ denotes that the Hamiltonian has acted completely. Transitions between the levels are performed in the MPO by the non-trivial blocks. Contracting the MPO from left to right, one can never go down a level. 

Written out in full, the Hamiltonian is given by 

$$
H = \sum_ {i} (D _ {i} + C _ {i} B _ {i + 1} + C _ {i} A _ {i + 1} B _ {i + 2} + C _ {i} A _ {i + 1} A _ {i + 2} B _ {i + 3} + \dots). \tag {12}
$$

This shows that any Hamiltonian with exponentially decaying interactions can be efficiently represented by an MPO of this form. Moreover, other decay profiles can often be very well approximated by this type of MPO [10, 11]. 

# 2.4 Examples

It is instructive to give a few examples of Hamiltonians written in this form, partly because we will use these examples as benchmark cases in Sec. 8. The nearest-neighbour transverse-field Ising model is defined by the Hamiltonian 

$$
H _ {\text {ising,nn}} = - \sum_ {i} Z _ {i} Z _ {i + 1} + h \sum_ {i} X _ {i} \sim \left( \begin{array}{c c c} \mathbb {I} & - Z & h X \\ & 0 & Z \\ & & \mathbb {I} \end{array} \right). \tag {13}
$$

In this case, the diagonal A block is zero and the dimension of the middle level is $\chi = 1$ . This Hamiltonian can be extended with long-range exponentially-decaying interactions by including an entry on the diagonal 

$$
H _ {\text {ising,lr}} = - \sum_ {i <   j} \lambda^ {j - i - 1} Z _ {i} Z _ {j} + h \sum_ {i} X _ {i} \sim \left( \begin{array}{c c c} \mathbb {I} & - Z & h X \\ & \lambda \mathbb {I} & Z \\ & & \mathbb {I} \end{array} \right), \quad \lambda <   1. \tag {14}
$$

Another paradigmatic example is the Heisenberg spin-1/2 chain, represented as 

$$
H _ {\text { heisenberg,nn }} = \sum_ {i} S _ {i} ^ {\alpha} S _ {j} ^ {\alpha} \sim \left( \begin{array}{c c c} \mathbb {I} & S ^ {\alpha} & 0 \\ & 0 & S ^ {\alpha} \\ & & \mathbb {I} \end{array} \right). \tag {15}
$$

Here, the spin operators are $S ^ { \alpha } = ( S ^ { x } , S ^ { y } , S ^ { z } )$ , such that the blocks have dimension $\chi = 3 . { } ^ { 4 }$ A next-nearest-neighbour $J _ { 1 } { - } J _ { 2 }$ spin-1/2 chain is given by 

$$
H _ {\text { heisenberg,nnn }} = J _ {1} \sum_ {i} S _ {i} ^ {\alpha} S _ {i + 1} ^ {\alpha} + J _ {2} \sum_ {i} S _ {i} ^ {\alpha} S _ {i + 2} ^ {\alpha} \sim \left( \begin{array}{c c c c} \mathbb {I} & S ^ {\alpha} & 0 & 0 \\ & 0 & \mathbb {I} & J _ {1} S ^ {\alpha} \\ & 0 & 0 & J _ {2} S ^ {\alpha} \\ & & & \mathbb {I} \end{array} \right), \tag {17}
$$

where the tensor I in the A block again represents the direct product of two unit matrices, 

$$
\mathbb {I} = - \Bigg \rangle -. \tag {18}
$$

Finally, we give an example of a two-dimensional system, which we have wrapped onto a cylinder such that the model can be reformulated as a one-dimensional system. The transversefield Ising model on a square lattice formulated on a cylinder of circumference $L _ { y }$ with spiral boundary conditions is given by 

$$
H _ {\text {ising,cylinder}} = - \sum_ {i} Z _ {i} Z _ {i + 1} - \sum_ {i} Z _ {i} Z _ {i + L _ {y}} + h \sum_ {i} X _ {i} \sim \left( \begin{array}{c c c c c c c} \mathbb {I} & - Z & 0 & 0 & \dots & 0 & h X \\ & 0 & \mathbb {I} & 0 & \dots & 0 & Z \\ & 0 & 0 & \mathbb {I} & \dots & 0 & 0 \\ & \vdots & & & \ddots & & \vdots \\ & 0 & & & \dots & \mathbb {I} & 0 \\ & 0 & & & \dots & 0 & Z \\ & & & & & & \mathbb {I} \end{array} \right). \tag {19}
$$

$$
\left( \begin{array}{c c c c c} \mathbb {I} & S ^ {x} & S ^ {y} & S ^ {z} & 0 \\ & 0 & 0 & 0 & S ^ {x} \\ & 0 & 0 & 0 & S ^ {y} \\ & 0 & 0 & 0 & S ^ {z} \\ & & & & \mathbb {I} \end{array} \right).
$$

When encoding SU(2) symmetry, however, we cannot split up the MPO into spin components (which break SU(2) invariance) and we have to keep the above form with the Sα tensor defined as 

$$
\boxed { \begin{array}{c c} \hline & \end{array} } = \boxed {S} \xrightarrow {\alpha} \boxed {S}, \quad S ^ {\alpha} = \boxed {S} \xrightarrow {\alpha}. \tag {16}
$$

Here, the leg denoted by α transforms under the spin-1 representation of SU(2). 

# 2.5 Powers of MPOs

This MPO representation of Hamiltonians is convenient for expressing powers of the Hamiltonian, and evaluating e.g. the variance or higher-order cumulants of the Hamiltonian with respect to a given MPS. 

We start by rewriting the Hamiltonian in table form: 

<table><tr><td></td><td>(1)</td><td>(2)</td><td>(3)</td></tr><tr><td>(1)</td><td><eq>\mathbb{I}</eq></td><td>C</td><td>D</td></tr><tr><td>(2)</td><td></td><td>A</td><td>B</td></tr><tr><td>(3)</td><td></td><td></td><td><eq>\mathbb{I}</eq></td></tr></table>

We can now represent $H ^ { 2 }$ , the product of this Hamiltonian with itself, as a sparse MPO of the form 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td><td>(1,3)</td><td>(2,1)</td><td>(2,2)</td><td>(2,3)</td><td>(3,1)</td><td>(3,2)</td><td>(3,3)</td></tr><tr><td>(1,1)</td><td><eq>\mathbb{I}</eq></td><td>C</td><td>D</td><td>C</td><td>CC</td><td>CD</td><td>D</td><td>DC</td><td>DD</td></tr><tr><td>(1,2)</td><td></td><td>A</td><td>B</td><td></td><td>CA</td><td>CB</td><td></td><td>DA</td><td>DB</td></tr><tr><td>(1,3)</td><td></td><td></td><td><eq>\mathbb{I}</eq></td><td></td><td></td><td>C</td><td></td><td></td><td>D</td></tr><tr><td>(2,1)</td><td></td><td></td><td></td><td>A</td><td>AC</td><td>AD</td><td>B</td><td>BC</td><td>BD</td></tr><tr><td>(2,2)</td><td></td><td></td><td></td><td></td><td>AA</td><td>AB</td><td></td><td>BA</td><td>BB</td></tr><tr><td>(2,3)</td><td></td><td></td><td></td><td></td><td></td><td>A</td><td></td><td></td><td>B</td></tr><tr><td>(3,1)</td><td></td><td></td><td></td><td></td><td></td><td></td><td><eq>\mathbb{I}</eq></td><td>C</td><td>D</td></tr><tr><td>(3,2)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>A</td><td>B</td></tr><tr><td>(3,3)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td><eq>\mathbb{I}</eq></td></tr></table>

(21) 

Here we have used a particular notation for combining the blocks: we take the operator product (composition) on the physical legs and a direct product on the virtual legs. For example: 

$$
= \boxed {A A} = \boxed {A} \tag {22}
$$

Upon computing the triple-layer transfer matrix associated to taking the expectation value of $H ^ { 2 }$ with respect to an injective MPS, the diagonal blocks I in the above form will give rise to an eigenvalue 1, which will have an (algebraic) multiplicity of 4. For reasons to be explained in Section 4, this will decompose into a one-dimensional eigenspace that does not couple to the boundary conditions, and a three-dimensional generalised eigenspace, or thus a threedimensional Jordan block, giving rise to terms scaling as a second order polynomial of N upon taking the Nth power. It therefore represents a second-degree MPO and its expectation value can be evaluated using the methods of Refs. [19, 22]. Again, we can understand this MPO as a finite-state machine 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/6c97332287bc4ecf4d11af74ff6c246520e40e6f5e179001bab944c167b11e17.jpg)


where we have omitted the operators denoting the different transitions in the graph (they can be read off from the table). The structure of this MPO is best understood by decomposing it into two parts, i.e. the disconnected terms and the connected terms. The former are the terms that are the direct product of single actions of the Hamiltonians that do not overlap, and in the diagram they are obtained by passing through levels (1,3) or $^ { ( 3 , 1 ) }$ . Indeed, the meaning of these levels is that one of the Hamiltonians has already acted, whereas the second one has not. The connected terms are the ones where the two Hamiltonian operators overlap. For example, jumping from (1,1) or (1,2) immediately to (2,3) means that the two Hamiltonian operators overlap on one site, and similarly for the jump from (1,1) or (2,1) to (3,2). All the other connected terms pass through level (2,2), which denotes that both Hamiltonian operators are acting simultaneously, and therefore this level has a bond dimension $\chi ^ { 2 }$ . 

# 3 From powers of the Hamiltonian to extensive MPOs

Let us now investigate how to approximate the exponential of the Hamiltonian in terms of MPOs. We take a generic spin-chain Hamiltonian $H = \textstyle \sum _ { i } h _ { i }$ , with $h _ { i }$ the (quasi) local hamiltonian operator acting on sites i, i + 1, . . . , which can be represented as an MPO of the form in Eq. (20). We wish to approximate 

$$
\mathrm{e} ^ {\tau H} = \mathbb {I} + \tau H + \frac {\tau^ {2}}{2} H ^ {2} + \frac {\tau^ {3}}{6} H ^ {3} + \dots , \tag {23}
$$

where we assume that τ is a small parameter. Naively, one could try to use the above representation of $H ^ { n }$ to approximate the exponential. Adding different powers of H is, however, an ill-defined operation in the thermodynamic limit because the norms of these different terms scale with different powers of system size. Therefore, applying a sum of different powers of H to a given state |Ψ〉, would yield a state 

$$
\mathrm{e} ^ {\tau H} | \Psi \rangle \approx \sum_ {i = 0} ^ {N} | \Psi_ {i} \rangle , \quad \langle \Psi_ {i} | \Psi_ {i} \rangle \propto N ^ {i}, \tag {24}
$$

which cannot be normalized in the thermodynamic limit.5 

Instead, an appropriate MPO representation of $\mathsf { e } ^ { \tau H }$ requires a size-extensive approach. Therefore, we introduce a transformation that maps a given power of H to a size-extensive operator, yielding an n’th order approximation for $\mathrm { e } ^ { \tau H }$ . We start at first order. Given the finite-state machine representation of H, the transformation can be visualized as 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/b0cda7dc4e95f100f99774bc773edaac9d1cbaab1bb33590b03869a9e920b988.jpg)


I.e., instead of falling onto the level ‘3’ in the MPO for the Hamiltonian, we go back to level ‘1’ and we omit level ‘3’ from the MPO. In addition, we multiply with the appropriate factor τ. In table form, this gives rise to 

<table><tr><td></td><td>(1)</td><td>(2)</td></tr><tr><td>(1)</td><td><eq>\mathbb{I} + \tau D</eq></td><td>C</td></tr><tr><td>(2)</td><td><eq>\tau B</eq></td><td>A</td></tr></table>

(26) 

which serves as a first-order approximation of the time evolution operator $\mathrm { e } ^ { \tau H }$ , as introduced in Ref. [7]. In the absence of any Jordan blocks, this operator is size-extensive: upon applying this MPO to a normalizable state, it returns a normalizable state. It is also size-extensive in another sense: it contains all disconnected higher-order terms in the expansion (with correct prefactor), i.e. higher order terms in which different actions of the Hamiltonian do not overlap. Indeed, if we write out the MPO from Eq. (26) in orders of τ we obtain 

$$
\mathbb {I} + \tau \sum_ {i} h _ {i} + \tau^ {2} \sum_ {i <   j, \text {disc}} h _ {i} h _ {j} + \tau^ {3} \sum_ {i <   j <   k, \text {disc}} h _ {i} h _ {j} h _ {k} + \dots , \tag {27}
$$

where the second and third sum runs over all terms for which the $h _ { i }$ do not overlap. 

This transformation can be extended to second order, where we have to include the terms where two actions of the Hamiltonian overlap. These are contained within the MPO representation of $H ^ { 2 }$ [Eq. (21)], so this is the starting point. The level (1,3) encodes the situation where one action of the Hamiltonian has been applied, while the other Hamiltonian can be recognized in the subblock 

<table><tr><td></td><td>(1,3)</td><td>(2,3)</td><td>(3,3)</td></tr><tr><td>(1,3)</td><td><eq>\mathbb{I}</eq></td><td>C</td><td>D</td></tr><tr><td>(2,3)</td><td></td><td>A</td><td>B</td></tr><tr><td>(3,3)</td><td></td><td></td><td><eq>\mathbb{I}</eq></td></tr></table>

This level (1,3) therefore encodes a disconnected term in $H ^ { 2 }$ and should be immediately mapped back to the starting state (1,1). The (3,1) level is completely equivalent to the (1,3) level, and should also be mapped back to the starting state (1,1). In practice this can be done by taking the columns $^ { ( 1 , 3 ) }$ and $^ { ( 3 , 1 ) }$ in $H ^ { 2 }$ , multiplying by $\frac { \tau } { 2 } .$ , and adding them to the first column. Afterwards both columns are removed, and we end up with the MPO: 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td><td>(2,1)</td><td>(2,2)</td><td>(2,3)</td><td>(3,2)</td><td>(3,3)</td></tr><tr><td>(1,1)</td><td><eq>\mathbb{I} + \tau D</eq></td><td>C</td><td>C</td><td>CC</td><td>CD</td><td>DC</td><td>DD</td></tr><tr><td>(1,2)</td><td><eq>\frac{\tau}{2}B</eq></td><td>A</td><td></td><td>CA</td><td>CB</td><td>DA</td><td>DB</td></tr><tr><td>(2,1)</td><td><eq>\frac{\tau}{2}B</eq></td><td></td><td>A</td><td>AC</td><td>AD</td><td>BC</td><td>BD</td></tr><tr><td>(2,2)</td><td></td><td></td><td></td><td>AA</td><td>AB</td><td>BA</td><td>BB</td></tr><tr><td>(2,3)</td><td></td><td></td><td></td><td></td><td>A</td><td></td><td>B</td></tr><tr><td>(3,2)</td><td></td><td></td><td></td><td></td><td></td><td>A</td><td>B</td></tr><tr><td>(3,3)</td><td></td><td></td><td></td><td></td><td></td><td></td><td><eq>\mathbb{I}</eq></td></tr></table>

In terms of the finite-state machine, one can think of this operation as follows 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/06c78e294cd473c4beb5fe532c32fa035c177766be352b4e6ff988ca849f209f.jpg)


The (3,3) level represents the state where both Hamiltonians were applied. Because we have already filtered out the disconnected contributions in the previous step, this state now only contains the connected second-order contributions! Similar to the (1,3) case, we can take the (3,3) column, this time multiply by $\frac { \tau ^ { 2 } } { 2 }$ , and add it to the first column. Then remove the (3,3) row and column: 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td><td>(2,1)</td><td>(2,2)</td><td>(2,3)</td><td>(3,2)</td></tr><tr><td>(1,1)</td><td><eq>\mathbb{I} + \tau D + \frac{\tau^{2}}{2}DD</eq></td><td>C</td><td>C</td><td>CC</td><td>CD</td><td>DC</td></tr><tr><td>(1,2)</td><td><eq>\frac{\tau}{2}B + \frac{\tau^{2}}{2}DB</eq></td><td>A</td><td></td><td>CA</td><td>CB</td><td>DA</td></tr><tr><td>(2,1)</td><td><eq>\frac{\tau}{2}B + \frac{\tau^{2}}{2}BD</eq></td><td></td><td>A</td><td>AC</td><td>AD</td><td>BC</td></tr><tr><td>(2,2)</td><td><eq>\frac{\tau^{2}}{2}BB</eq></td><td></td><td></td><td>AA</td><td>AB</td><td>BA</td></tr><tr><td>(2,3)</td><td><eq>\frac{\tau^{2}}{2}B</eq></td><td></td><td></td><td></td><td>A</td><td></td></tr><tr><td>(3,2)</td><td><eq>\frac{\tau^{2}}{2}B</eq></td><td></td><td></td><td></td><td></td><td>A</td></tr></table>

(31) 

or in terms of a finite-state machine, we take the transformation 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/51b300e7667fc9c2417899ef9e606639dc9e31ef9bc19d1bf812d3865420658c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/fbadc91935739c4c6a057e2ca1385e8ac7be3739cf30cb794a3c98faad721594.jpg)


The above MPO now gives an approximation of $\mathrm { e } ^ { \tau H }$ that captures all second-order terms exactly. Moreover, just as before, due to its size extensivity, it contains all higher-order terms that consist of disconnected first- and second-order parts. 

This construction can be generalized to any order by the same idea, and the algorithm can be found in Alg. 1. 

# Algorithm 1 Pseudocode for constructing the N’th order time evolution MPO

1: Inputs $\hat { H } , N , \tau$ 

2: $O \gets \hat { H } ^ { N }$ ▷ multiply the hamiltonian N times with itself 

3: for $a \in [ 1 , N ]$ do 

4: $P \gets$ permutations of $( 1 , 1 , . . . , 1 , 3 , 3 , . . . , 3 )$ (3 occurs a times) 

5: for $b \in P$ do 

6: 号 $\begin{array} { r } { O [ : , 1 ] = O [ : , 1 ] + \tau ^ { a } \frac { ( N - a ) ! } { N ! } O [ : , b ] } \end{array}$ 

7: Remove row and column b 

In this section, we have explained our construction in terms of a single MPO tensor, but the construction is easily extended for systems with a non-trivial unit cell. For finite systems, one should impose the correct left and right boundary conditions: 

$$
L = \boxed {1} \boxed {0} \boxed {\dots} \boxed {0}, \quad R = \boxed { \begin{array}{c} 1 \\ \hline 0 \\ \hline \dots \\ \hline 0 \end{array} }. \tag {33}
$$

# 4 Exact compression steps

The operator we arrived at in the previous section is essentially an operator-valued block matrix, a matrix where the entries correspond to operators. It is possible to multiply these by scalar-valued block matrices and in particular we can left and right multiply with the matrix 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td><td>(2,1)</td><td>(2,2)</td><td>(2,3)</td><td>(3,2)</td></tr><tr><td>(1,1)</td><td>1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(1,2)</td><td></td><td><eq>1/\sqrt{2}</eq></td><td><eq>1/\sqrt{2}</eq></td><td></td><td></td><td></td></tr><tr><td>(2,1)</td><td></td><td><eq>1/\sqrt{2}</eq></td><td><eq>-1/\sqrt{2}</eq></td><td></td><td></td><td></td></tr><tr><td>(2,2)</td><td></td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td>(2,3)</td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td>(3,2)</td><td></td><td></td><td></td><td></td><td></td><td>1</td></tr></table>

(34) 

to obtain the MPO 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td><td>(2,1)</td><td>(2,2)</td><td>(2,3)</td><td>(3,2)</td></tr><tr><td>(1,1)</td><td><eq>\mathbb{I} + \tau D + \frac{\tau^{2}}{2} DD</eq></td><td>C</td><td></td><td>CC</td><td>CD</td><td>DC</td></tr><tr><td>(1,2)</td><td><eq>\tau B + \frac{\tau^{2}}{2}(DB + BD)</eq></td><td>A</td><td></td><td>CA+AC</td><td>CB+AD</td><td>DA+BC</td></tr><tr><td>(2,1)</td><td><eq>\tau B + \frac{\tau^{2}}{2}(DB - BD)</eq></td><td></td><td>A</td><td>CA-AC</td><td>CB-AD</td><td>DA-BC</td></tr><tr><td>(2,2)</td><td><eq>\frac{\tau^{2}}{2}BB</eq></td><td></td><td></td><td>AA</td><td>AB</td><td>BA</td></tr><tr><td>(2,3)</td><td><eq>\frac{\tau^{2}}{2}B</eq></td><td></td><td></td><td></td><td>A</td><td></td></tr><tr><td>(3,2)</td><td><eq>\frac{\tau^{2}}{2}B</eq></td><td></td><td></td><td></td><td></td><td>A</td></tr></table>

(35) 

Given the boundary conditions at the left boundary [Eq. 33], there is no way to reach level $^ { ( 2 , 1 ) }$ . The corresponding entry in the left environments will always be zero and the corresponding row/column can therefore be safely removed. 

Another way to see this compression is to look at the graphical representation of the original MPO, and noting the symmetry: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/98dcccaeefd215d7b93ec6d4dd5d2dcc1e5fd86faf1c959baca506c62fa8cc41.jpg)


The transitions from (1,1) to (1,2) and from (1,1) to (2,1) are completely equivalent, we can therefore deform the diagram without changing the MPO. Simply add all arrows that leave the (2,1) node to the (1,2) node, and remove the (2,1) node. 

A similar observation holds for the (2,3) and (3,2) nodes: all operators that follow the node before arriving at $^ { ( 1 , 1 ) }$ are the same! We can redirect all arrows that point to $^ { ( 3 , 2 ) }$ , point them at $^ { ( 2 , 3 ) }$ and remove the node $^ { ( 3 , 2 ) }$ . Equivalently, a similar basis transformation as in Eq. 34 will eliminate the transition from one of the (2,3) (3,2) levels back to (1,1). Given the right boundary condition [Eq. 33], the right environment will be zero for that level, and the corresponding row/column can be removed. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/185e63c2e68c7f080e6bf0bd8a88e1b75a8018ab58f00befdcbfced00f91c935.jpg)


We eventually end up with the following operator: 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td><td>(2,2)</td><td>(2,3)</td></tr><tr><td>(1,1)</td><td><eq>\mathbb{I} + \tau D + \frac{\tau^{2}}{2} DD</eq></td><td>C</td><td>CC</td><td>CD+DC</td></tr><tr><td>(1,2)</td><td><eq>\tau B + \frac{\tau^{2}}{2}(DB+BD)</eq></td><td>A</td><td>CA+AC</td><td>CB+AD+DA+BC</td></tr><tr><td>(2,2)</td><td><eq>\frac{\tau^{2}}{2} BB</eq></td><td></td><td>AA</td><td>AB+BA</td></tr><tr><td>(2,3)</td><td><eq>\frac{\tau^{2}}{2}B</eq></td><td></td><td></td><td>A</td></tr></table>

(38) 

which represents a compressed version of the original second-order MPO in Eq. (31). 

This exact compression step can be generalized to the general n’th order MPOs, see Alg. 2. 


Algorithm 2 Pseudocode incorporating exact compression


<table><tr><td colspan="3">1: O ← Alg. 1</td></tr><tr><td colspan="3">2: for c ∈ possible levels in O do</td></tr><tr><td>3:</td><td><eq>s_c \leftarrow</eq> Sort the 1’s in c to the front</td><td></td></tr><tr><td>4:</td><td><eq>s_r \leftarrow</eq> Sort the 3’s in c to the front</td><td></td></tr><tr><td>5:</td><td><eq>n_1 \leftarrow</eq> the number of 1’s in c</td><td></td></tr><tr><td>6:</td><td><eq>n_3 \leftarrow</eq> the number of 3’s in c</td><td></td></tr><tr><td>7:</td><td>if <eq>n_3 \leq n_1</eq> &amp; <eq>s_c \neq c</eq> then</td><td>▷ Equivalent column</td></tr><tr><td>8:</td><td><eq>O[s_c, :] = O[s_c, :] + O[c, :]</eq></td><td>▷ Add row c to row <eq>s_c</eq></td></tr><tr><td>9:</td><td>Remove row and column c</td><td></td></tr><tr><td>10:</td><td>if <eq>n_3 &gt; n_1</eq> &amp; <eq>s_r \neq c</eq> then</td><td>▷ Equivalent row</td></tr><tr><td>11:</td><td><eq>O[:, s_r] = O[:, s_r] + O[:, c]</eq></td><td>▷ Add column c to column <eq>s_r</eq></td></tr><tr><td>12:</td><td>Remove row and column c</td><td></td></tr></table>

# 5 Incorporating higher-order terms

At this point, we have found an MPO expression for $\mathsf { e } ^ { \tau H }$ that is correct up to a given order $n ,$ but which also contains all disconnected higher-order terms that can be decomposed into smaller-order factors. Yet we can still incorporate more higher-order terms in the MPO without changing the bond dimension. Starting from the first-order MPO in Eq. 26, it was indeed noticed in Ref. [7] that the second-order term with Hamiltonians only overlapping on a single site can be readily included in the MPO. In this section, we show that our construction of the N th order MPO can be similarly extended to contain all terms of order N +1, for which at least two out of the N + 1 composed Hamiltonian terms are such that one term ends on the same site as the other one starts. Terms that cannot be captured in this way are those that contain the composition of $N + 1$ contributions of the A block on a given site. 

The easiest way to understand this procedure is by studying the finite-state machines that generate $\hat { H } ^ { N }$ and $\hat { H } ^ { N + 1 }$ , before turning it into the extensive zero-degree MPO that represents the exponential of $\hat { H }$ and applying any of the compression steps. Consider a certain path through the finite-state machine of $\hat { H } ^ { N }$ , that starts at the N -tuple $( 1 , 1 , \ldots , 1 )$ and ends at the N -tuple $( 3 , 3 , \ldots , 3 )$ . Within this path, we also want to systematically encode contributions coming from $\hat { H } ^ { N + 1 }$ , namely contributions where at one particular site, which corresponds to one particular segment along the path, one term of the Hamiltonian stops and another term starts. This corresponds to having an extra mode 1 in the incoming tuple, and an extra mode 3 in the outgoing tuple. Note that this extra 1 and 3 can appear everywhere in the tuple, i.e. one operator term is forced to stop on this site and another term is started. The entry of the MPO representation of $\hat { H } ^ { N + 1 }$ corresponding to these extended tuples exactly contains the correct contribution to make this happen, so that we can add this contribution to the existing entry of $\hat { H } ^ { N }$ for the original values of these tuples. The other segments of the path through the finite state machine do not need to be changed. If for example the extra 1 and 3 appear on the same position in the extended tuples, this corresponds to an extra contribution of the on-site operator encoded in D, but that is certainly not the only possible contribution. We have to account for the fact that, after transforming this MPO into the extensive MPO using Algorithm 1, this contribution will be given the prefactor $\tau ^ { N } / N !$ , whereas it should have a prefactor $\tau ^ { N + 1 } / ( N + 1 ) !$ , which we can easily compensate by attributing it a proper factor. Furthermore, by adding an extra 1 and 3 in the incoming and outgoing tuples at all possible positions, identical configurations with multiple 1s and 3s in the extended tuples will be counted several times, namely exactly as many times as the number of 1s or 3s that appear in these extended tuples. This too is simply corrected for by dividing with these factors. 

One final remark is that we only want to add these additional contributions to paths in the finite-state machine that already encode terms of $\hat { H } ^ { N }$ where the N different factors overlap. This requires in particular that there are no 1s appearing in the right tuple, as this would indicate that some factors of $\hat { H } ^ { N }$ still have to start. Furthermore, if the left tuple would only contain 1s and one or more ${ 3 s , }$ this corresponds to a contribution where some terms have already ended, and would count as a lower order contribution when building the extensive MPO. Hence, aside from the $( 1 , 1 , \ldots , 1 )$ tuple, all left tuples should contain one or more values 2. 

The resulting algorithm is represented by the pseudocode in Alg. 3. The application of this extension step to the case $N = 2 .$ , followed by the transformation to the extensive MPO (Algorithm 1) and the compression step described in the previous section (which remains valid), gives rise to the following $\mathsf { M P O } ^ { 6 }$ 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td><td>(2,2)</td></tr><tr><td>(1,1)</td><td><eq>\mathbb{I} + \tau D + \frac{\tau^{2}}{2}DD + \frac{\tau^{3}}{6}DDD</eq></td><td>C</td><td><eq>CC + \tau \{CCD\}</eq></td></tr><tr><td>(1,2)</td><td><eq>\tau(B + \tau \{DB\} + \frac{\tau^{2}}{2}\{BDD\})</eq></td><td>A</td><td><eq>2\{AC\} + \tau(\{ACD\} + \{BCC\})</eq></td></tr><tr><td>(2,2)</td><td><eq>\frac{\tau^{2}}{2}(BB + \tau\{BBD\})</eq></td><td></td><td><eq>AA + \tau\{AAD\} + 2\tau\{ABC\}</eq></td></tr><tr><td>(2,3)</td><td><eq>\frac{\tau^{2}}{2}B + \frac{\tau^{3}}{6}(2\{BD\} + BD)</eq></td><td></td><td><eq>\frac{\tau}{3}(2\{AC\} + AC)</eq></td></tr></table>

<table><tr><td></td><td>(2,3)</td></tr><tr><td>(1,1)</td><td><eq>2\{CD\} + \tau\{CDD\}</eq></td></tr><tr><td>(1,2)</td><td><eq>2(\{BC\} + \{AD\}) + \tau(2\{BCD\} + \{ADD\})</eq></td></tr><tr><td>(2,2)</td><td><eq>2\{AB\} + \tau\{ABD\} + \frac{\tau}{3}\{BBC\}</eq></td></tr><tr><td>(2,3)</td><td><eq>A + \frac{\tau}{3}(2\{AD\} + AD + 2\{BC\} + BC)</eq></td></tr></table>


Algorithm 3 Pseudocode for the extension step


1: Inputs $\hat{H}, N, dt$ 2: $O \leftarrow \hat{H}^N$ 3: for $a, b \in$ possible levels in $O$ & $1 \notin b$ do ▷ Incorporate higher order corrections
4: if $2 \notin a$ & $3 \in a$ , skip
5: for $c, d \in [1:N+1]$ do
6: $a_e =$ insert a 1 at position $c$ in $a$ 7: $b_e =$ insert a 3 at position $d$ in $b$ 8: $n_1 =$ the number of 1's in $a_e$ 9: $n_3 =$ the number of 3's in $b_e$ 10: $O[a, b] = O[a, b] + H^{N+1}[a_e, b_e]\tau \frac{N!}{(N+1)!n_1 n_3}$ 11: Apply algorithm 2 

# 6 Approximate compressions

There is one more possible compression for an Nth order ${ \mathrm { M P O } } ,$ similar in spirit to the previous extension step. This compression step is only accurate up to order N , and it therefore slightly lowers the precision of the extended MPO. We will again illustrate the method starting from the second-order ${ \mathrm { M P O } } ,$ and then extend it to arbitrary order. 

The essential observation is that the levels (12) and (23) in the second order MPO are similar. The diagonal elements $O _ { 2 } [ ( 1 2 ) , ( 1 2 ) ]$ and $O _ { 2 } [ ( 2 3 ) , ( 2 3 ) ]$ are equal in the lowest order in τ. Furthermore, the transition from (12) to (11) and from (23) to (11) are also related. The lowest order of $O _ { 2 } [ ( 2 3 ) , ( 1 1 ) ]$ equals the lowest order of $O _ { 2 } [ ( 1 2 ) , ( 1 1 ) ]$ , multiplied with an extra factor of $\frac { \tau } { 2 }$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/625d2956ac1649798c5fb52847ff5b4c36cb607bf49e6ed9ea350a029f9758c7.jpg)


(40) 

This means that one can add $\frac { \tau } { 2 }$ times the (23) column to the (21) column and remove the (23) level, and the resulting MPO will also be accurate up to second order! The second-order MPO now becomes 

<table><tr><td></td><td>(1,1)</td><td>(1,2)</td></tr><tr><td>(1,1)</td><td><eq>\mathbb{I} + \tau D + \frac{\tau^{2}}{2!}D^{2} + \frac{\tau^{3}}{3!}D^{3}</eq></td><td><eq>C + \tau\{CD\} + \frac{\tau^{2}}{2!}\{CDD\}</eq></td></tr><tr><td>(1,2)</td><td><eq>\tau(B + \tau\{BD\} + \frac{\tau^{2}}{2!}\{BDD\})</eq></td><td><eq>A + \tau\{AD\} + \frac{\tau^{2}}{2!}\{ADD\} + \tau(\{CB\} + \tau\{CBD\})</eq></td></tr><tr><td>(2,1)</td><td><eq>\frac{\tau^{2}}{2!}(BB + \tau\{BBD\})</eq></td><td><eq>\tau(\{AB\} + \tau\{ABD\}) + \frac{\tau^{2}}{2!}\{CBB\}</eq></td></tr></table>

<table><tr><td></td><td>(2,2)</td></tr><tr><td>(1,1)</td><td><eq>CC + \tau\{CCD\}</eq></td></tr><tr><td>(1,2)</td><td><eq>2(\{AC\} + \tau\{ACD\}) + \tau\{CCB\}</eq></td></tr><tr><td>(2,1)</td><td><eq>AA + \tau\{AAD\} + 2\tau\{ACB\}</eq></td></tr></table>

Once again we can generalize this step to any order, as described in Alg. 4. 


Algorithm 4 Pseudocode for the approximate compression step


1: Apply algorithm 3
2: for $a \in$ possible levels in O & $1 \notin a$ do
3: $n_{1} = \text{the number of 3's in } a$ 4: b = replace all 3's with 1's in a
5: $O[:, b] = O[:, b] + O[:, a] \tau^{n_{1}} \frac{(N - n_{1})!}{N!}$ 6: remove level a 

# 7 Numerical compression

In the previous three sections, we have provided analytical techniques for compressing and extending our construction for approximating $\mathrm { e } ^ { \tau H }$ as an MPO. However, we can also compress the MPO numerically using singular-value decompositions. The idea behind this is that we interpret the MPO as a regular MPS with two physical legs, and truncate with respect to the 2-norm for states. This procedure should be taken with care, because we are working with a norm that is not suitable for operators, and should maybe only be used in cases where we can do exact compressions (for which the singular values are exactly zero, and it doesn’t matter which norm is taken). 

We can use this numerical compression for checking whether we have found all exact compression steps. If we do this on the uncompressed MPOs from Sec. 3, we observe that we indeed find a number of exact zero singular values in the MPO, corresponding to the analytical compression steps that we have identified above. After having done these analytical compressions, however, we find that the MPO cannot be compressed further. This shows that we have found all possible exact compressions. 

# 8 Benchmarks

# 8.1 Precision of nth order MPO

Let us first illustrate the precision of our MPO construction. Therefore, we first optimize an MPS ground-state approximation $\left| \Psi _ { 0 } \right.$ of a given Hamiltonian H in the thermodynamic limit and subsequently evaluate 

$$
p = \left| \lambda - i E _ {0} \delta t \right|, \quad \lambda = \lim _ {N \rightarrow \infty} \frac {1}{N} \log \left\langle \right. \Psi_ {0} \left. \right| U (\delta t) \left| \right. \Psi_ {0} \left. \right\rangle , \tag {42}
$$

where λ can be evaluated directly in the thermodynamic limit (see Sec. 2) and $E _ { 0 }$ is the groundstate energy per site. In this set-up, we make sure that the MPS $\left| \Psi _ { 0 } \right.$ is approximating the true ground state quasi-exactly – in practice, we just take very large bond dimension – such that p is indeed measuring errors in the MPO approximation U(δt) for the time-evolution operator. 

In Fig. 1 we plot this quantity as a function of δt for both the nth-order MPO without extensions and approximate compressions and the extended and compressed MPO, each time for different orders. We find that the error has the expected scaling as a function of δt, showing that our MPO construction is correct up to a given order. We observe that the approximate compression and extension steps give rise to more precise MPOs, although the bond dimension is smaller. This shows that it is always beneficial to work with these MPOs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/8505e8079cde683e3514056dfc0086a7f007524e350f8f90d2f8ce1c77559abe.jpg)



Figure 1: Precision of the nth order operator (as measured by Eq. 42) for the SU(2) symmetric spin-1 Heisenberg model, plotted as a function of the time step. We find the expected power-law behavior, where different orders directly correspond to different powers (the nth order operator has an error scaling as n + 1). The open circles show the results from only using exact compression steps, while the filled circles were obtained using both the approximate compression and extension. Note that the fluctuations around $1 0 ^ { - 1 2 }$ are due to limited numerical accuracy.


# 8.2 Efficiency

After having showed that our construction works as intended, we now show that it is actually efficient to use higher-order MPOs in practical MPS time-evolution algorithms. Let us therefore take the Hamiltonian of the two-dimensional transverse-field Ising model on a finite cylinder with spiral boundary conditions [Eq. (19)], find an MPS ground-state, perform a spin flip in the middle of the cylinder and time-evolve the state. This is the typical set-up for evaluating a spectral function. We time-evolve for a total time $T = 1$ with different times steps $\delta t ,$ where we approximate the time-evolution operator $\mathrm { e } ^ { i H \delta t }$ by MPOs of different orders. In each time step, we perform a variational sweeping optimization of the new time-evolved MPS and keep the bond dimension fixed. After the time evolution we evaluate the fidelity per site with respect to a benchmark time-evolved state (which was obtained by the algorithm based on the time-dependent variational principle (TDVP) [23] with time step $\delta t = 0 . 0 0 0 1 )$ : 

$$
f = \frac {1}{N} \log \left\langle \Psi_ {\text {bench}} (T) | \Psi_ {n} (T) \right\rangle , \tag {43}
$$

with N the number of sites. 

In the first panel of Fig. 2 we plot this fidelity density as a function of time step, showing that we indeed find higher precision with higher-order MPOs and that the error scales with the correct power of $\delta t$ . Note that the first-order MPO is exactly the same as the $W _ { I I }$ operator from Ref. [7]. Curiously, we find that the error for the second-order MPO scales according to a third-order ${ \mathrm { M P O } } ,$ but this is not generically true and depends on the particular Hamiltonian. 

In the second panel, we show the computational time as a function of the fidelity density, showing how much time is needed to reach a certain accuracy. This plot clearly shows that it is beneficial to go beyond the first-order MPO. For general models, we expect that we can obtain better fidelity at the same computational cost by using higher order methods. The extraordinary performance of the second-order MPO in this particular example originates from the fact that it is correct up to third-order, which is not expected in general. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/6502546feb966fbcd02024a0c7038d8d83dcbfb3c9da1ae88b426040562f7b58.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/518e54f4539d46de5fb771edf52e1173adfaec7041c1c5ddc108ba49cf8fa8a6.jpg)



Figure 2: Benchmark results for the transverse-field Ising model on a cylinder $( W = 4 )$ . In the left panel we plot the fidelity density [Eq. 43] as a function of time step. In the right panel we plot the fidelity density as a function of total simulation time.


# 8.3 Splitting schemes

There is a well known approach for generating higher-order time-evolution methods out of lower-order approximation schemes, by combining ingeniously chosen time steps [24]. Given a first order method, such as our time-evolution operator $O _ { 1 } ( t )$ , it can be combined with alternating timesteps $t _ { 1 } = ( 1 + i ) / 2$ and $t _ { 2 } = ( 1 - i ) / 2$ . The composite operator $O _ { 1 } ( t _ { 1 } ) O _ { 1 } ( t _ { 2 } )$ $= O _ { 2 } ( t _ { 1 } + t _ { 2 } )$ is then accurate up to second order [7]. In general, a second-order method and more than two time steps are required, in order to construct higher order schemes by combining only real time steps. This is also the basis behind higher order Suzuki-Trotter decompositions. 

In contrast to these splitting schemes, the construction of the N th order MPO $O _ { N }$ has a bond dimension as listed in the following table (where $\chi$ is the bond dimension of the A block in the Hamiltonian): 

<table><tr><td>Order</td><td>Bond dimension</td></tr><tr><td>1</td><td>$ 1 + \chi $</td></tr><tr><td>2</td><td>$ 1 + \chi + \chi^{2} $</td></tr><tr><td>3</td><td>$ 1 + 3\chi + \chi^{2} + \chi^{3} $</td></tr><tr><td>4</td><td>$ 1 + 5\chi + 4\chi^{2} + \chi^{3} + \chi^{4} $</td></tr></table>

Even in the case that we assume that our MPO operators are fully dense, the composition of N first-order operators will therefore always have a larger bond dimension than the construction we put forward. 

Furthermore, for splitting schemes including complex-valued time steps, the resulting operator will exponentially grow high energy contributions before exponentially suppressing them in a subsequent step, raising serious concerns about their stability. These splitting schemes may however be useful as a trade-off between CPU time and memory usage. A highorder time evolution operator corresponds to an MPO tensor with an exponentially large bond dimension. By combining a splitting scheme with the highest-order operator that can still reside in memory, one can push time evolution simulations to even higher levels of accuracy. 

# 8.4 Finite temperature

Our method can also be used to directly construct the finite temperature density matrix $e ^ { - \beta H }$ , at different orders of precision. We have calculated the free energy and energy density for different values of $\beta$ at different expansion orders for the spin- $- \frac 1 2$ XXZ model, directly in the thermodynamic limit (see Fig. 3). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/1bfed81994f53e7d6aedfca478e12bcd044bb7af023ec0571e2e0c5be1d7da4f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-03/68030de2-7892-47e8-ab73-7be10850bdf7/c6b19f2c147ab64726325ec73f8a6b95340c6ae00ad4c52a48f24b7520e456db.jpg)



Figure 3: Finite temperature results for the $\mathrm { s p i n } { - } \frac { 1 } { 2 }$ XXZ model in the thermodynamic limit. In the left panel we plot the energy density. The ground state energy density is indicated by a black dashed line. In the right panel we plot the free energy density.


This calculation is highly and straightforwardly parallelizable (at least on a shared-memory architecture), as it boils down to solving an iterative dominant eigenvalue problem of a blocksparse matrix. It is however fundamentally limited in the achievable $\beta .$ . At some crossover point (around $\beta \sim 2$ in this case) the error term will always start to dominate, and the results become wildly inaccurate. The best results will presumably be obtained by multiplying multiple density matrices at smaller $\beta$ (which can be calculated up to arbitrary precision). 

# 9 Conclusion and outlook

We have introduced a new way of approximating the time evolution operator as an MPO correctly up to arbitrary order in the time step. The algorithm is formulated in the language of Hamiltonians represented as first-degree MPOs and is directly compatible with spatial symmetries (in particular, translation invariance) and non-abelian on-site symmetries. While such a construction is interesting in its own right, we have demonstrated that a higher order scheme allows us to speed up MPO time-evolution simulations by an order of magnitude – for a detailed comparison between MPO schemes and TDVP algorithms, we refer the reader to Ref. [17]. The higher-order MPOs can be readily used in existing time-evolution algorithms, leading to immediate speedups. For the reader’s convenience, we have summarized the most useful MPO expressions in the Appendix. 

It would be interesting to explore the interplay between the approximate compression step from Sec. 6 and the extension step from Sec. 5. The compression step should in principle introduce errors of order $N + 1$ , while these are precisely the kind of terms we correctly try to incorporate in the extension step, and so in principle we would expect these steps to be at odds with each other. Nevertheless we observe that a combination of the two gives the best results, which is not yet fully understood. 

In principle it is clear how one can apply a very similar methodology to time-dependent Hamiltonians. For the example of periodic driving, it could allow us to construct the time evolution operator over an entire period at once. In turn, we would be able to analyze this operator with spectral methods, extracting information on the effective time-averaged operator, as an alternative to the more conventional perturbative expansion. 

In another direction, we expect that our MPO construction can be useful for optimizing other approximation schemes for the time-evolution operator. Notably, the use of efficient MPO representations can greatly benefit the classical optimization of quantum circuits for implementing dynamics on digital quantum simulators [25, 26]. 

# Acknowledgments

We would like to thank Bram Vanhecke and Frank Verstraete for earlier collaborations that inspired this work. 

Code availability The computer code can be found in the software package MPSKit.jl [27]. 

Funding information MV and JH have received support from the European Research Council (ERC) under the European Union’s Horizon 2020 program [Grant Agreement No. 715861 (ERQUAF)] and from the Research Foundation Flanders. LV is supported by the Research Foundation Flanders (FWO) via grant FWO20/PDS/115. IPM is supported by the National Science and Technology Council (NSTC) Grant Nos. 112-2811-M-007-044 and 113-2112-M-007-MY2. 

# A Explicit expressions

Here we recapitulate the expressions for the optimal first- and second-order MPOs. Starting from a Hamiltonian in MPO form 

<table><tr><td><eq>\mathbb{I}</eq></td><td><eq>C</eq></td><td><eq>D</eq></td></tr><tr><td></td><td><eq>A</eq></td><td><eq>B</eq></td></tr><tr><td></td><td></td><td><eq>\mathbb{I}</eq></td></tr></table>

the optimal first-order MPO is given by 

<table><tr><td>$ \mathbb{I} + \tau D + \frac{\tau^{2}}{2} D^{2} $</td><td>$ C + \tau\{CD\} $</td></tr><tr><td>$ \tau(B + \tau\{BD\}) $</td><td>$ A + \tau\{AD\} + \tau\{CB\} $</td></tr></table>

and the optimal second-order MPO is given by 

<table><tr><td><eq>\mathbb{I} + \tau D + \frac{\tau^{2}}{2}D^{2} + \frac{\tau^{3}}{6}D^{3}</eq></td><td><eq>C + \tau\{CD\} + \frac{\tau^{2}}{2}\{CDD\}</eq></td><td><eq>CC + \tau\{CCD\}</eq></td></tr><tr><td><eq>\tau(B + \tau\{BD\} + \frac{\tau^{2}}{2}\{BDD\})</eq></td><td><eq>A + \tau\{AD\} + \frac{\tau^{2}}{2}\{ADD\} + \tau(\{CB\} + \tau\{CBD\})</eq></td><td><eq>2(\{AC\} + \tau\{ACD\}) + \tau\{CCB\}</eq></td></tr><tr><td><eq>\frac{\tau^{2}}{2}(BB + \tau\{BBD\})</eq></td><td><eq>\tau(\{AB\} + \tau\{ABD\}) + \frac{\tau^{2}}{2}\{BBC\}</eq></td><td><eq>AA + \tau\{AAD\} + 2\tau\{ACB\}</eq></td></tr></table>

The expressions for the higher-order MPOs are too large to display on this page, and we advise to implement the generic algorithms from the main text. 

# References



[1] S. R. White, Density matrix formulation for quantum renormalization groups, Phys. Rev. Lett. 69, 2863 (1992), doi:10.1103/PhysRevLett.69.2863. 





[2] U. Schollwöck, The density-matrix renormalization group in the age of matrix product states, Ann. Phys. 326, 96 (2011), doi:10.1016/j.aop.2010.09.012. 





[3] G. Vidal, Efficient simulation of one-dimensional quantum many-body systems, Phys. Rev. Lett. 93, 040502 (2004), doi:10.1103/PhysRevLett.93.040502. 





[4] S. R. White and A. E. Feiguin, Real-time evolution using the density matrix renormalization group, Phys. Rev. Lett. 93, 076401 (2004), doi:10.1103/PhysRevLett.93.076401. 





[5] A. J. Daley, C. Kollath, U. Schollwöck and G. Vidal, Time-dependent density-matrix renormalization-group using adaptive effective Hilbert spaces, J. Stat. Mech.: Theor. Exp. 04, P04005 (2004), doi:10.1088/1742-5468/2004/04/P04005. 





[6] E. M. Stoudenmire and S. R. White, Minimally entangled typical thermal state algorithms, New J. Phys. 12, 055026 (2010), doi:10.1088/1367-2630/12/5/055026. 





[7] M. P. Zaletel, R. S. K. Mong, C. Karrasch, J. E. Moore and F. Pollmann, Time-evolving a matrix product state with long-ranged interactions, Phys. Rev. B 91, 165112 (2015), doi:10.1103/PhysRevB.91.165112. 





[8] J. Haegeman, J. I. Cirac, T. J. Osborne, I. Pižorn, H. Verschelde and F. Verstraete, Timedependent variational principle for quantum lattices, Phys. Rev. Lett. 107, 070601 (2011), doi:10.1103/PhysRevLett.107.070601. 





[9] B. Vanhecke, L. Vanderstraeten and F. Verstraete, Symmetric cluster expansions with tensor networks, Phys. Rev. A 103, L020402 (2021), doi:10.1103/PhysRevA.103.L020402. 





[10] B. Pirvu, V. Murg, J. I. Cirac and F. Verstraete, Matrix product operator representations, New J. Phys. 12, 025012 (2010), doi:10.1088/1367-2630/12/2/025012. 





[11] D. E. Parker, X. Cao and M. P. Zaletel, Local matrix product operators: Canonical form, compression, and control theory, Phys. Rev. B 102, 035147 (2020), doi:10.1103/PhysRevB.102.035147. 





[12] L. Vanderstraeten, J. Haegeman and F. Verstraete, Tangent-space methods for uniform matrix product states, SciPost Phys. Lect. Notes 7 (2019), doi:10.21468/SciPostPhysLectNotes.7. 





[13] F. Verstraete, J. J. García-Ripoll and J. I. Cirac, Matrix product density operators: Simulation of finite-temperature and dissipative systems, Phys. Rev. Lett. 93, 207204 (2004), doi:10.1103 PhysRevLett.93.207204. 





[14] F. Verstraete and J. I. Cirac, Renormalization algorithms for quantum-many body systems in two and higher dimensions, (arXiv preprint) doi:10.48550/arxiv.cond-mat/0407066. 





[15] M. Hauru, M. Van Damme and J. Haegeman, Riemannian optimization of isometric tensor networks, SciPost Phys. 10, 040 (2021), doi:10.21468/SciPostPhys.10.2.040. 





[16] B. Vanhecke, M. Van Damme, J. Haegeman, L. Vanderstraeten and F. Verstraete, Tangent-space methods for truncating uniform MPS, SciPost Phys. Core 4, 004 (2021), doi:10.21468/SciPostPhysCore.4.1.004. 





[17] S. Paeckel, T. Köhler, A. Swoboda, S. R. Manmana, U. Schollwöck and C. Hubig, Time-evolution methods for matrix-product states, Ann. Phys. 411, 167998 (2019), doi:10.1016/j.aop.2019.167998. 





[18] M. Stoudenmire, G. Evenbly, S. R. White and I. McCulloch, MPO-MPS multiplication: Density matrix algorithm, https://tensornetwork.org/mps/algorithms/denmat_mpo_mps. 





[19] L. Michel and I. P. McCulloch, Schur forms of matrix product operators in the infinite limit, (arXiv preprint) doi:10.48550/arXiv.1008.4667. 





[20] C. Hubig, I. P. McCulloch and U. Schollwöck, Generic construction of efficient matrix product operators, Phys. Rev. B 95, 035129 (2017), doi:10.1103/PhysRevB.95.035129. 





[21] G. M. Crosswhite and D. Bacon, Finite automata for caching in matrix product algorithms, Phys. Rev. A 78, 012356 (2008), doi:10.1103/PhysRevA.78.012356. 





[22] J. C. Pillay and I. P. McCulloch, Cumulants and scaling functions of infinite matrix product states, Phys. Rev. B 100, 235140 (2019), doi:10.1103/PhysRevB.100.235140. 





[23] J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken and F. Verstraete, Unifying time evolution and optimization with matrix product states, Phys. Rev. B 94, 165116 (2016), doi:10.1103/PhysRevB.94.165116. 





[24] E. Hairer, C. Lubich and G. Wanner, Geometric numerical integration, Springer, Berlin, Heidelberg, Germany, ISBN 9783662050187 (2002), doi:10.1007/978-3-662-05018-7. 





[25] M. S. J. Tepaske, D. Hahn and D. J. Luitz, Optimal compression of quantum manybody time evolution operators into brickwall circuits, SciPost Phys. 14, 073 (2023), doi:10.21468/SciPostPhys.14.4.073. 





[26] M. S. J. Tepaske, D. Hahn and D. J. Luitz, Optimal compression of quantum manybody time evolution operators into brickwall circuits, SciPost Phys. 14, 073 (2023), doi:10.21468 SciPostPhys.14.4.073. 





[27] M. Van Damme, M. Hauru, G. Roose, L. Devos, L. Vanderstraeten and J. Haegeman, MPSKit.jl, https://github.com/maartenvd/MPSKit.jl. 

