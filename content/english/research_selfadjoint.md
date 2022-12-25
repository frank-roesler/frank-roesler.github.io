---
title: "Research"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "english"
---

Computational Quantum Mechanics
-------------------------------

### Research article:
*   Frank R&ouml;sler; [On The Solvability Complexity Index for Unbounded Selfadjoint and Schr&ouml;dinger Operators.](https://link.springer.com/article/10.1007%2Fs00020-019-2555-x) *Integral Equations and Operator Theory*, (2019) 91:54.
(A Matlab implementation of the algorithm is available [here](https://github.com/frank-roesler/Work))

### Overview:
This work is concerned with the computational solution of the Schr&ouml;inger eigenvalue problem. The Schr&ouml;dinger equation is at the heart of the theory of Quantum Mechanics and its eigenfunctions and eigenvalues describe the bound states of quantum systems and their corresponding energy levels. In its dimensionless form, the problem reads: *find a function $\mathsf\psi$ and a complex number $\mathsf\lambda$ such that the equation*
$$
	\mathsf{-\Delta \psi + V\psi = \lambda\psi} \tag{1}
$$
*holds*. The scalar function $\mathsf V$ in the above equation describes the potential energy of the system (e.g. due to an electric field, in which the particle moves). In most practical applications, this equation cannot be solved analytically and numerical solution methods need to be applied. For potentials, which are both smooth and real-valued, there exist methods, which are efficient and mostly reliable (note however the issue of [spectral pollution](https://arxiv.org/pdf/math/0302145.pdf)). However, for complex-valued potentials, the computational eigenvalue problem becomes considerably more difficult and finding robust, reliable numerical algorithms that apply to a large class of potentials becomes a major challenge. In my article above, I contruct a new algorithm, which is taylored to precisely this situation. The procedure is guaranteed to converge to the correct solution of $(1)$ for any potential $\mathsf{V(x)}$, which decays to 0 as $\mathsf{|x|\to\infty}$.  
{{< figure src="/images/research/sch_gauss.png" link="/images/research/sch_gauss.png" >}}
The mathematical idea behind the algorithm is to replace the differential operator $\mathsf{-\Delta+V}$ by a matrix approximation and compute its [pseudospectrum](/research_pseudospectrum), whose singularities coincide with the eigenvalues. The convergence proofs use the so-called [essential numerical range](https://www.sciencedirect.com/science/article/abs/pii/S0022123620300525) to control spectral pollution. The figure above shows a sample output of the algorithm, implemented in MATLAB, for a complex valued potential. The algorithm correctly approximates the continuous spectrum on the ppositive real axis, as well as two eigenvalues at approximately $\mathsf{2\pm 3.9i}$.











