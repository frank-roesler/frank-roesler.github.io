---
title: "Research"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "english"
---

Computing Bound States in Relativistic Quantum Mechanics
--------------------------------------------------------

### Research article:
*   Frank R&ouml;sler, Christiane Tretter; [Computing Klein-Gordon Eigenvalues,](https://arxiv.org/abs/2210.12516) *preprint*.
(A Matlab implementation of the algorithm is available [here](https://github.com/frank-roesler/spectral_klein_gordon). A recent presentation on the topic is also [available online](/images/slides_kg.pdf))

### Overview:
The computational spectral problem in quantum mechanics is becoming more and more well understood. Particularly the numerical complexity of the classical (nonrelativistic) theory that goes back to Schr&ouml;dinger and Heisenberg has been studied in a number of different situations by a number of different authors (including myself and my collaborators), see [here](/research_selfadjoint), [here](/research_periodic) or [here](/research_resonances). This has not been the case for the *relativistic* theory, which is concerned with particles whose velocities approach the speed of light. The underpinning of relativistic quantum mechanics is formed by two equations of motion, the *Dirac equation* and the *Klein-Gordon equation*. Either of these equations replaces the classical Schr&ouml;dinger equation in relativistic settings (depending on whether the particle considered has nonzero [spin](https://en.wikipedia.org/wiki/Spin_(physics))).  
Our new research work above concerns the computation of eigenvalues for the Klein-Gordon equation. This equation governs the evolution of relativistic, spinless particles, most famously the [*Higgs Boson*](https://en.wikipedia.org/wiki/Higgs_boson), which was discovered by the ATLAS and CMS experiments at the Large Hadron Collider (LHC) at CERN and earned Peter Higgs and Fran&ccedil;ois Englert a Nobel Prize in Physics in 2013. For a particle of mass $\mathsf{m}$ and charge $\mathsf{e}$, interacting with an electric field $\mathsf{\varphi}$, the equation reads
$$\mathsf{\left( - \Bigl( -i \hbar\frac \partial {\partial t} - e \varphi
\Bigr)^2 - \hbar^2c^2 \Delta + m^2c^4 \right) \psi = 0,}$$
where $\mathsf{c}$ denotes the speed of light and $\mathsf{\hbar}$ is Planck's constant. If we separate time by setting $\mathsf{\psi(x,t) =: e^{i\lambda/\hbar t} u(x)}$, $\mathsf{x\in\mathbb R^d}$, $\mathsf{t\in\mathbb R}$, normalize $\mathsf{c}$ to $\mathsf{1}$, let $\mathsf{V}$ be the multiplication operator by $\mathsf{e\varphi}$, the above equation simplifies to
$$ \mathsf{(-\Delta+m^2)u = (V-\lambda)^2u. \tag{1}}$$
Solving the above equation amounts to finding a pair $\mathsf{\lambda\in\mathbb C}$, $\mathsf{u\in L^2(\mathbb R^2)}$ such that $(1)$ holds, where $\mathsf V$ is a known function. This problem is reminiscent of the Schr&ouml;dinger eigenvalue problem, with one major difference: the potential function $\mathsf V$ and the spectral parameter $\mathsf{\lambda}$ enter quadratically, rather than linearly! This renders the problem not only physically, but also mathematically non-classical: one cannot simply transfer results from the nonrelativistic theory, but needs to come up with novel solution techniques.  
{{< figure src="/images/research/kg_result.png" link="/images/research/kg_result.png" >}}
We developed a numerical algorithm, based on an abstract, mathematical understanding of equation $(1)$, which automatically computes its eigenvalues $\lambda$ in the complex plane, given a set of point values $\mathsf{V(x)}$ of the potential. The algorithm can be implemented and run on a laptop and its output is guaranteed to converge to the correct set of eigenvalues of the Klein-Gordon equation. The mathematical strategy behind our algorithm is to find an operator related to $\mathsf{(-\Delta+m^2)-(V-\lambda)^2}$, which can be well approximated by a finite size matrix $\mathsf{K(\lambda)}$, and identify the approximate eigenvalues as the poles of its Fredholm inverse $\mathsf{(I+K(\lambda))^{-1}}$. The above figure shows the output of the algorithm for a smoothed square well potential. As can be seen, it returns 8 approximate eigenvalues in the complex plane, 6 of which are non-real. This non-reality of the eigenvalues, even when the potential is real-valued is a truly relativistic effect: it would be impossible for the Schr&ouml;dinger equation. In the physics community, this is known as the [*Schiff-Snyder-Weinberg Effect*](https://journals.aps.org/pr/abstract/10.1103/PhysRev.57.315).













