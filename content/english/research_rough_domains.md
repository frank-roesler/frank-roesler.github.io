---
title: "Research"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "english"
---

Computing Vibration Modes of Fractal Drums
------------------------------------------

### Research article:
*	Frank R&ouml;sler, Alexei Stepanenko; [Computing Eigenvalues of the Laplacian on Rough Domains](https://arxiv.org/abs/2104.09444), *Preprint*, arXiv:2104.09444.  
	(A Matlab implementation of the algorithm is available [here](https://github.com/frank-roesler/PixelSpectra))

### Overview:
In this recent research project my collaborator Alexei Stepanenko and I studied computational aspects of a classical problem in spectral theory: Computing the eigenmodes of the 2d surface of a drum. Those are given by the eigenfunctions and eigenvalues of the Laplace operator with zero boundary conditions: 
$$\mathsf{-\Delta u = \lambda u \quad\text{on the drum }\mathcal{O},} \tag{1}$$
where $u$ models the vertical displacement of the drum's membrane and $\lambda$ its frequency. If the shape of the drum is highly symmectric (e.g. circular), this problem can be [solved explicitly](https://en.wikipedia.org/wiki/Vibrations_of_a_circular_membrane?wprov=sfti1). However, if the drum has a more complicated shape, the problem becomes very difficult and can in general only be solved numerically. In our research article we study a numerical procedure for solving precisely this problem. We demonstrate that even for drums whose boundary is a *fractal* (i.e. varies wildly on every length scale, no matter how small) the eigenmodes $\mathsf u$ and eigenfrequencies $\mathsf\lambda$ can be computed reliably. Our method relies on increasingly fine pixelations $\mathsf{\mathcal{O}\_n}$ of the original drum, as in the next figure.
{{< figure src="/images/research/koch.png" link="https://frank-roesler.github.io/images/research/koch.png" >}}
These pixelated domains are suitable for a solution of $(1)$ based on the [finite element method](https://en.wikipedia.org/wiki/Finite_element_method?wprov=sfti1). We provide a freely available implementation of our algorithm that can indeed be applied off-the-shelf to 2-dimensional domains of any shape. As an illustrative example, the figure below shows the first 12 vibration eigenmodes of a drum that has the shape of the United Kingdom.
<!-- {{< figure src="/images/research/uk_pixel.png" link="https://frank-roesler.github.io/images/research/uk_pixel.png" >}} -->

{{< figure src="/images/research/uk_modes_lr.png" link="https://frank-roesler.github.io/images/research/uk_modes2.png" >}}



