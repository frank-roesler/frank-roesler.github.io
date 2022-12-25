---
title: "Research"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "english"
---

A Strange Vertex Condition Coming from Nowhere
----------------------------------------------

### Research article:
*   Frank R&ouml;sler; [A Strange Vertex Condition Coming from Nowhere.](https://epubs.siam.org/doi/abs/10.1137/20M1322194) *SIAM J. Math. Anal.*, 53(3), 3098--3122, 2021

### Overview:
This is one of my single author papers in the field of asymptotic analysis. It is concerned with the combined effect of two domain properties: *thin geometry* and *perforation*. 

**Thin Geometry:**
Consider the Laplacian $\mathsf{-\Delta}$ on a domain $\mathsf{\Omega_\epsilon\subset\mathbb R^d}$ and suppose that $\mathsf{\Omega_\epsilon}$ approximates a graph $\mathsf\Gamma$ as $\mathsf{\epsilon\to0}$, as in the following figure.
{{< figure src="/images/research/graph1.png" link="/images/research/graph1.png" >}}
One can prove[^1] that as $\mathsf{\epsilon\to 0}$ the Laplacian on $\mathsf{\Omega_\epsilon}$ converges to a Laplacian on the graph with certain matching conditions at the vertices (similar to the [Kirchhoff conditions](https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws)). The precise nature of these vertex conditions depends on the relative scaling of the edge neighbourhoods and the vertex neighbourhoods of the domain $\mathsf{\Omega_\epsilon}$.

**Perforation:**
A *perforated domain* is a domain, from which a periodic arrangement of balls is removed, where both the distance $\mathsf \epsilon$ and the radii $\mathsf{r_\epsilon}$ of the balls are much smaller than the diameter of the domain, as shown in the following figure (for more details, see my [corresponding post](/research_perfdom)).
{{< figure src="/images/research/perfdom.png" link="/images/research/perfdom.png" >}}
This perforation results in a domain $\mathsf{\Omega_\epsilon}$, on which a partial differential equation can be studied (depending on the equation, this can be used to model materials with a fine microstructure). 
It has been shown[^2] that
$$
	\mathsf{-\Delta_{\Omega_\epsilon} \to -\Delta_{\Omega}+\mu} \quad \text{ as }\quad\mathsf{\epsilon\to 0},
$$
where $\mathsf\mu$ is a positive constant, which has been dubbed the "strange term".[^2]

**Combination:**
I my article, I studied the combined effect of thin, graph-like geometry and fine perforation, i.e. I studied domains which approximate a graph and in addition are finely perforated.
{{< figure src="/images/research/perfdom2.png" link="/images/research/perfdom2.png" >}}
This combination leads to a striking observation: as in the case of purely thin geometry, a limit operator on the graph $\mathsf\Gamma$ can be found, which, on each edge of $\mathsf\Gamma$ is given by a shifted second derivative $\mathsf{-\frac{d^2}{dx^2}+\mu}$. On its own, this result would be only moderately surprising, but as it turns out the strange term $\mathsf\mu$ *also* appears in the *vertex conditions* of the limit problem! More precisely, the limit problem of the equation $\mathsf{(-\Delta+z)u=f}$ on $\mathsf{\Omega_\epsilon}$ turns out to be
$$
	\begin{cases}
	\mathsf{(-\Delta+z+\mu) u = f} &\textsf{ on }\mathsf \Gamma\\\\
			\qquad \mathsf{\sum_{e\ni v} u'\_e(v) = (z+\mu) \frac{|V|}{|\Omega_0|} u(v)}, &\textsf{ at each vertex }\mathsf v,
	\end{cases}
$$
that is, the strange term $\mathsf\mu$ (which describes the "density" of the perforation) can be used as a tuning parameter to modify the Kirchhoff vertex conditions!

[^1]: [P. Exner and O. Post. *Convergence of spectra of graph-like thin manifolds.* J. Geom. Phys. , 54(1) :77-115, 2005]
[^2]: [Cioranescu, Doina, and Francois Murat. *A strange term coming from nowhere.* Topics in the mathematical modelling of composite materials. Birkh&auml;user, Boston, MA, 1997. 45-93.]
