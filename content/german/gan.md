---
title: "Hobbyprojekte"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "german"
---

Wie lässt man einen Computer halluzinieren?
-------------------------------------------

Ein sogenanntes *Generative adversarial Network (GAN)* ist eine Kombination aus zwei neuronalen Netzen (engl. "Generator" und "Discriminator"), die kombiniert werden, um Bilder eines bestimmten Typs zu erzeugen. Die beiden Netze konkurrieren gegeneinander: Zunächst erzeugt der Generator ein Bild aus einem Zufallsvektor, danach vergleicht der Discriminator das erzeugte Bild mit einem vorher festgelegten Trainingsdatensatz.  
Der Discriminator wird darauf trainiert, die künstlichen Bilder von den echten Bildern (d.h. dem Trainingsdatensatz) zu unterscheiden, während der Generator darauf trainiert wird, den Discriminator zu täuschen. Dieser gemeinsame Trainingsprozess treibt den Generator dazu, Bilder zu erzeugen, die dem Trainingsdatensatz immer ähnlicher sehen.  
Die oben beschriebene Methode wurde in der Arbeit [[Radford, Metz, & Chintala (2015)]](https://arxiv.org/pdf/1511.06434.pdf) beschrieben. Inspiriert durch [diese PyTorch-Implementierung](https://github.com/pytorch/examples/tree/main/dcgan) habe ich ein GAN darauf trainiert, Bilder von Katzen zu erzeugen. Die Resultate erinnern an Gemälde des Impressionismus und sind unten zu sehen.

{{< figure src="/images/gan_cats.png" link="https://frank-roesler.github.io/images/gan_cats.png">}}