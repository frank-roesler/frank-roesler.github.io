---
title: "Research"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "english"
---

Pseudospectra of Non-Hermitian Hamiltonians
-------------------------------------------

### Research article:
*   Patrick W. Dondl, Patrick Dorey, Frank R&ouml;sler; [A Bound on the Pseudospectrum for a Class of Non-normal Schr&ouml;dinger Operators.](https://academic.oup.com/amrx/article/2017/2/271/2769429?searchresult=1) *Appl. Math. Res. Express* 2016.

### Overview:
This was the first project of my PhD. The work is concerned with the [Schr&ouml;dinger equation](/research_selfadjoint) with complex-valued potentials. In general, the eigenvalues (or the *spectrum*) of such operators could be any complex number, however, some potentials exhibit special symmetries (called PT-symmetry), which constrain their eigenvalues to lie on the real axis. An example is the *imaginary cubic oscillator*, defined as
$$
	\mathsf{H = -\frac{d^2}{dx^2} + ix^3} \quad \textsf{ on }\quad \mathsf{L^2(\mathbb R)}
	\tag{1}
$$
Indeed, this operator can be shown to be PT-symmetric and its eigenvalues are purely real, as the following numerical plot illustrates.
{{< figure src="/images/research/ho_spectrum.png" link="/images/research/ho_spectrum.png" >}}
Operators like $(1)$ are of interest in the theory of [Non-Hermitian Quantum Mechanics](https://en.wikipedia.org/wiki/Non-Hermitian_quantum_mechanics)[^1]. This example shows that, given only the spectrum of an operator, it is impossible to decide whether its potential is real-valued or not (i.e. whether the operator is *selfadjoint* or not). This motivates the introduction of a finer indicator, called the *pseudospectrum*. The pseudospectrum is the collection of level sets of the function $\mathsf{R(z) := \\|(H-z)^{-1}\\|}$. It can be shown that the eigenvalues of $\mathsf H$ are precisely the poles of $\mathsf{R(z)}$ and that for *real-valued* potentials $\mathsf{R(z)}$ is determined completely by the distance of $\mathsf z$ to the set of eigenvalues of $\mathsf H$ (accordingly, the level sets of $\mathsf{R(z)}$ are circles around the eigenvalues). For *non-real* potential, however, the pseudospectrum can deviate very strongly from the set of eigenvalues. Indeed, the next plot shows a contour plot of $\mathsf{R(z)}$ for the imaginary cubic oscillator $(1)$.
{{< figure src="/images/research/ho_pseudospectrum.png" link="/images/research/ho_pseudospectrum.png" >}}
As the figure shows, the function $\mathsf{R(z)}$ may assume very large values far away from the eigenvalues. This behaviour would be impossible for a hamiltonian with a real-valued potential.  
Other examples from the literature show that the pseudospectrum of non-hermitian operators can indeed look arbitrarily chaotic. Our own contribution in the paper mentioned above was to prove that for a class of examples such as the imaginary cubic oscillator the situation is not all bad: Even though the pseudospectrum is not simply given by the distance to the spectrum, we prove certain bounds on $\mathsf{R(z)}$, which imply that on any half plane which extends to the left in the complex plane, the function *is* determined, to some extent, by the eigenvalues in that half plane. Even though suggested by the numerical figure above, this result is highly nontrivial and not guaranteed except for a very limited class of operators.



[^1]: [Bender, Carl M. *Making sense of non-Hermitian Hamiltonians.* Reports on Progress in Physics 70.6 (2007): 947.]
