---
title: "Research"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "english"
---

Homogenisation in Perforated Domains
------------------------------------

### Research article:
*   Patrick Dondl, Kirill Cherednichenko, Frank R&ouml;sler; [Norm-Resolvent Convergence in Perforated Domains.](https://content.iospress.com/articles/asymptotic-analysis/asy1481) *Asymptotic Analysis*, vol. 110, no. 3-4, pp. 163-184, 2018

### Overview:
A *perforated domain* is a domain, from which a periodic arrangement of balls is removed, where both the distance $\mathsf \epsilon$ and the radii $\mathsf{r_\epsilon}$ of the balls are much smaller than the diameter of the domain, as shown in the following figure.
{{< figure src="/images/research/perfdom.png" link="/images/research/perfdom.png" >}}
This perforation results in a domain $\mathsf{\Omega_\epsilon}$, on which a partial differential equation can be studied (depending on the equation, this models materials with a fine microstructure). 
It has been shown[^1] that (provided the correct scaling between $\mathsf \epsilon$ and $\mathsf{r_\epsilon}$ is chosen) the Laplacian of a function $\mathsf{-\Delta_{\Omega_\epsilon} u}$ on $\mathsf{\Omega_\epsilon}$ converges to a limit, which is *not* merely a Laplacian on the original domain $\mathsf{\Omega}$, but a shifted version $\mathsf{(-\Delta_\Omega + \mu)u}$, where $\mathsf\mu$ is a constant depending on the size of the holes. Several authors have dubbed this constant "A strange term coming from nowhere"[^2], because it is not present for any positive value of $\mathsf\epsilon$. 

While the convergence $\mathsf{-\Delta_{\Omega_\epsilon} \to -\Delta_{\Omega}+\mu}$ on individual functions has been established already in the 1960s, the question of whether the *spectrum* of $\mathsf{-\Delta_{\Omega_\epsilon}}$ converges to the spectrum of $\mathsf{-\Delta_{\Omega}+\mu}$ had been open until we proved it in the above article in 2018. In fact, we show not only convergence of spectra, but an even stronger statement, which has further implications for the asymptotic behaviour of the time-dependent problem $\mathsf{\partial_t u(t,x) = \Delta u(t,x)}$ on perforated domains. Such time dependent equations are used in applications for modelling the diffusion of gases and heat flow.







[^1]: [V. A. Marchenko & E. Ya. Khruslov. *Boundary-value problems with fine-grained boundary [in Russian]*, Mat. Sb. (N.S.), 1964, Volume 65(107), Number 3, 458-472]
[^2]: [Cioranescu, Doina, and Francois Murat. *A strange term coming from nowhere.* Topics in the mathematical modelling of composite materials. Birkh&auml;user, Boston, MA, 1997. 45-93.]