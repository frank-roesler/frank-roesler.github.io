---
title: "Forschung"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "german"
---

Berechnung von Schwingungsmoden fraktaler Trommeln  
----------------------------------------------------

### Forschungsartikel:
* Frank Rösler, Alexei Stepanenko; [Computing Eigenvalues of the Laplacian on Rough Domains](https://www.ams.org/journals/mcom/2024-93-345/S0025-5718-2023-03827-3/home.html), *Math. Comp.* 93 (2024), 111–161.  
(gefördert durch das Forschungs- und Innovationsprogramm Horizont 2020 der Europäischen Union im Rahmen des Marie Skłodowska-Curie-Stipendiums Nr. 885904)  
(Eine MATLAB-Implementierung des Algorithmus ist [hier verfügbar](https://github.com/frank-roesler/PixelSpectra))

### Überblick:
In diesem aktuellen Forschungsprojekt haben mein Kollege Alexei Stepanenko und ich rechnerische Aspekte eines klassischen Problems der Spektraltheorie untersucht: die Berechnung der Eigenschwingungen einer zweidimensionalen Trommeloberfläche. Diese sind gegeben durch die Eigenfunktionen und Eigenwerte des Laplace-Operators mit homogenen Randbedingungen:
$$\mathsf{-\Delta u = \lambda u \quad\text{auf der Trommel }\mathcal{O}} \tag{1}$$
wobei $u$ die vertikale Auslenkung der Membran und $\lambda$ deren Frequenz beschreibt. Ist die Form der Trommel stark symmetrisch (z. B. kreisförmig), lässt sich dieses Problem [explizit lösen](https://en.wikipedia.org/wiki/Vibrations_of_a_circular_membrane?wprov=sfti1). Bei komplexeren Formen wird das Problem jedoch sehr schwierig und kann im Allgemeinen nur numerisch behandelt werden.

In unserem Artikel stellen wir ein numerisches Verfahren zur Lösung genau dieses Problems vor. Wir zeigen, dass selbst bei Trommeln mit einem *fraktalen* Rand (d. h. mit extrem unregelmäßiger Geometrie auf allen Längenskalen) die Eigenfunktionen $\mathsf u$ und Eigenfrequenzen $\mathsf\lambda$ zuverlässig berechnet werden können. Unsere Methode basiert auf immer feineren Pixelierungen $\mathsf{\mathcal{O}_n}$ der ursprünglichen Trommel, siehe folgende Abbildung:
{{< figure src="/images/research/koch.png" link="https://frank-roesler.github.io/images/research/koch.png" >}}

Diese pixelbasierten Gebiete eignen sich hervorragend für eine numerische Lösung von Gleichung (1) mittels der [Finite-Elemente-Methode](https://en.wikipedia.org/wiki/Finite_element_method?wprov=sfti1). Wir stellen eine frei verfügbare Implementierung unseres Algorithmus zur Verfügung, der direkt auf zweidimensionale Domänen beliebiger Form angewendet werden kann.

Als anschauliches Beispiel zeigt die folgende Abbildung die ersten zwölf Schwingungsmoden einer Trommel in der Form des Vereinigten Königreichs:

{{< figure src="/images/research/uk_modes2.png" link="https://frank-roesler.github.io/images/research/uk_modes2.png" >}}
