---
title: "Forschung"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "german"
---

Rekonstruktion eines Potentials aus seinen Streudaten  
-----------------------------------------------------

### Forschungsartikel:  
* Jonathan Ben-Artzi, Marco Marletta und Frank Rösler. [On the complexity of the inverse Sturm-Liouville problem](https://msp.org/paa/2023/5-4/p03.xhtml), *Pure and Applied Analysis 5, 895-925 (2023)*.  
(unterstützt durch das Horizon 2020 Forschungs- und Innovationsprogramm der Europäischen Union im Rahmen des Marie Skłodowska-Curie-Stipendiums Nr. 885904.)  
(Eine Matlab-Implementierung des Algorithmus ist [hier](https://github.com/frank-roesler/inverse_SCI) verfügbar.)

### Überblick:  
Das sogenannte *Spektralproblem* für einen [Hamilton-Operator](/research_selfadjoint)  
$$
\mathsf{-\frac{d^2}{dx^2}+q}
$$
besteht darin, dessen Spektrum bei gegebener Potentialfunktion $\mathsf{q}$ zu bestimmen. In bestimmten Fällen (z.B. wenn $\mathsf{q(x)=x^2}$ das Potential des [harmonischen Oszillators](https://de.wikipedia.org/wiki/Harmonischer_Oszillator) ist oder die Gleichung auf einem beschränkten Intervall $\mathsf{[a,b]}$ betrachtet wird) ist das Spektrum durch die Menge der *Eigenwerte* gegeben, also alle Punkte $\mathsf{\lambda}$, für die  
$$
\mathsf{\bigg(-\frac{d^2}{dx^2}+q\bigg)\psi = \lambda\psi}
$$
für eine Funktion $\mathsf{\psi\in L^2(\mathbb R)}$ gilt, die man *Eigenfunktion* nennt.

Das *inverse Spektralproblem* fragt nun umgekehrt: Kann man aus dem Spektrum des Hamilton-Operators das Potential $\mathsf{q}$ rekonstruieren? Tatsächlich ist dies möglich – wenn zusätzlich die $\mathsf{L^2}$-Normen der zugehörigen Eigenfunktionen bekannt sind. Genauer gesagt: Sind zwei unendliche Zahlenfolgen $\mathsf{\lambda_n}$ und $\mathsf{\\|\psi_n\\|\_{L^2}}$ gegeben mit  
$$
\mathsf{\lambda_n^{\frac12} \sim n + \frac{\omega}{\pi n}},
$$
$$
\mathsf{\frac{1}{\\|\psi_n\\|\_{L^2}^2} \sim \frac{2}{\pi}},
$$
für $\mathsf{n\to\infty}$, so ist das Potential $\mathsf q$ eindeutig durch diese Folgen bestimmt.[^1] Die Folgen $\mathsf{\lambda_n}$, $\mathsf{\\|\psi_n\\|\_{L^2}}$ werden üblicherweise als *Spektraldaten* bezeichnet.

Eine schwierigere Frage ist, ob man die Funktion $\mathsf q$ *algorithmisch* aus diesen zwei Folgen berechnen kann. Da ein Computer zu jedem Zeitpunkt nur mit einer endlichen Anzahl von Zahlen arbeiten kann, lässt sich das Potential *nicht exakt* aus den unendlichen Spektraldaten rekonstruieren. Stattdessen müssen die Daten abgeschnitten werden, also nur endliche Teilfolgen  
$$
\mathsf{\lambda_1,\dots,\lambda_N}, \quad \mathsf{\\|\psi_1\\|\_{L^2},\dots,\\|\psi_N\\|\_{L^2}}
$$
verwendet werden. Daraus ergibt sich ein approximatives Potential $\mathsf{q_N}$, das bei wachsendem $\mathsf{N}$ gegen das ursprüngliche Potential $\mathsf q$ konvergieren soll.  
In unserem Forschungsartikel untersuchen wir genau, wie das Potential aus solchen abgeschnittenen Teilfolgen rekonstruiert werden kann und wie sich der durch das Abschneiden verursachte Fehler kontrollieren lässt. Wir konstruieren explizit einen Algorithmus, der das Potential aus den abgeschnittenen Spektraldaten berechnet. Die folgende Abbildung zeigt Beispielresultate für drei verschiedene Potentiale $\mathsf{q^{(1)}}$, $\mathsf{q^{(2)}}$, $\mathsf{q^{(3)}}$. Die erste Zeile zeigt die Resultate für eine Abschneidelänge von $\mathsf{N=9}$, die zweite für $\mathsf{N=29}$. Wie man sieht, verbessert sich die Approximation mit wachsendem $\mathsf N$.

{{< figure src="/images/research/inverse_results.png" link="https://frank-roesler.github.io/images/research/inverse_results.png" >}}

[^1]: [Freiling-Yurko. *Inverse Sturm-Liouville Problems and Their Applications.* Nova Science Pub Inc., 2001]
