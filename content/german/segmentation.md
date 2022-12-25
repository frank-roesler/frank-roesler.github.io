---
title: "Hobbyprojekte"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "german"
---

Bildsegmentierung mit FCNs
--------------------------

Auf meiner [GitHub-Seite](https://github.com/frank-roesler/Image_Segmentation)
habe ich nun eine "fully convolutional" (FCN) Version von Alexnet (siehe https://arxiv.org/abs/1411.4038) in Pytorch implementiert. Der Code in diesem Paket hat vor allem ein Ziel: Einfachheit. Erstens erkennt das Modell nur eine Art von Objekt (plus Hintergrund), zweitens sind alle Funktionen und Klassen explizit und transparent; nichts ist vortrainiert. Ziel ist es, die Grundlagen verständlich zu machen

Dieser Code wurde an dem ["Penn-Fudan"-Fußgänger-Datensatz](https://www.seas.upenn.edu/~jshi/ped_html/) und dem [COCO-Datensatz](https://cocodataset.org/) getestet (siehe Bilder). Obwohl die Trainingsdatensätze klein sind, liefert das Modell akzeptable Ergebnisse. Genauere Resultate können durch komplexere Netzwerkarchitekturen erzielt werden (z.B. transfer learning mit vortrainiertem ResNet-50).

{{< figure src="/images/seg2.png" class="center">}}
{{< figure src="/images/seg3.png" class="center">}}
{{< figure src="/images/seg4.png" class="center">}}  
(Bilder stammen aus dem COCO-Datensatz: https://cocodataset.org/)
