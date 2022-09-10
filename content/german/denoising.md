---
title: "Miscellaneous"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "german"
---

Rauschunterdrückung in Messdaten
--------------------------------

Methoden des maschinellen Lernens können in der Signalverarbeitung eingesetzt werden, um die Qualität von verrauschten Signalen zu verbessern. Ein Beispiel, das ich in PyTorch programmiert habe, legt nahe, dass diese Methode für Spektrosopie-Signale gut funktioniert. Solche Signale haben typischerweise [Lorentz-Form](https://de.wikipedia.org/wiki/Lorentzkurve). Reale Messdaten sind allerdings nie identisch mit der glatten Kurve, die die Theorie vorhersagt, sondern vielmehr verrauscht, wie das folgende Bild qualitativ zeigt (dunkel = Realteil des komplexen Signals, hell = Imaginärteil). Oft ist das Rauschen so stark, dass einzelne Peaks nicht mehr mit bloßem Auge zu erkennen sind.

{{< figure src="/images/denoising/gt.png" link="https://frank-roesler.github.io/images/denoising/gt.png" >}}

Synthetisch generierte Daten wie die obigen eignen sich, um ein neuronales Netz zu trainieren: Das Netz nimmt ein verrauschtes Signal entgegen und wird darauf trainiert, das Original (engl. “Ground Truth”) zu reproduzieren. Die Trainingsarchitektur ist im nächsten Bild veranschaulicht.

{{< figure src="/images/denoising/cnn.png" link="https://frank-roesler.github.io/images/denoising/cnn.png" >}}

Mein Code ist als Google Colab Notebook offen verfügbar:  [`my_notebook`](https://colab.research.google.com/drive/1wEljoKaEuV4WR-MtUWSFjBejXchJ5oJh?usp=sharing). Jeder kann den Code herunterladen oder (mit einem Google-Account) in der Cloud ausführen.
Die folgenden Bilder veranschaulichen das Ergebnis eines colvolutional Netzes mit 6 Layers, das mit MSE-Loss-Funktion trainiert wurde. Offensichtlich rekonstruiert das neuronale Netz die Originaldaten gut, wie das Schaubild des Approximationsfehlers (rechts unten) zeigt.

{{< figure src="/images/denoising/comparison.png" link="https://frank-roesler.github.io/images/denoising/comparison.png" >}}

Dieses Projekt ist Teil einer größeren Kollaboration (mehr Details gibt es in unserem kürzlichen [Abstract](https://www.ismrm.org/index.php?gf-download=2022%2F08%2FISMRM-Diff_Workshop_2022_ADoering.pdf&form-id=1341&field-id=16&hash=66281781958d5482202467a2c3b303a1a7d9c32327990656deee3fa54f9af27b)). Eine jüngere Version des Python Codes gibt es auf meiner GitHub-Seite: https://github.com/frank-roesler/MRspecNET (dieser Code ist noch work in progress).