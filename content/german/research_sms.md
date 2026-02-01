---
title: "Forschung"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "german"
---

Erzeugung neuer RF-Pulse für die MR-Spektroskopie  
--------------------------------------------------

Dies ist laufende Forschung. Das Code-Repository in der aktuellen Form ist hier verfügbar: https://github.com/frank-roesler/Inverse_Bloch

### Überblick:  
Simultaneous Multi-Slice (SMS) ist eine etablierte Technik in der Magnetresonanztomographie, die es ermöglicht, die Messzeit drastisch zu verkürzen, indem gleichzeitig Bilddaten aus mehreren Schichten im Körper des Patienten aufgenommen werden. Die Nutzung von SMS erfordert die parallele Anregung mehrerer Gewebeschichten mit demselben Hochfrequenz-(RF)-Impuls (die Grundlagen habe ich [hier](/bloch) erklärt).

Im verwandten Bereich der [Magnetresonanzspektroskopie](https://de.wikipedia.org/wiki/Magnetresonanzspektroskopie) ist SMS deutlich weniger etabliert, und neue Pulssequenzen befinden sich noch im Forschungsstadium. Ich bin an einem Forschungsprojekt zusammen mit [André Döring](https://www.epfl.ch/labs/metmrs/research/members/andre-doring-phd/) und [Lijing Xin](https://people.epfl.ch/lijing.xin?lang=en) an der EPFL Lausanne beteiligt, das darauf abzielt, SMS zu nutzen, um die Messzeit in der Magnetresonanz-Spektroskopieabbildung zu verbessern. Die Erzeugung von Multischicht-RF-Pulsen in der Spektroskopie ist deutlich komplexer als in der klassischen Bildgebung und erfordert ein sorgfältiges Abwägen verschiedener Einschränkungen.

Zu diesem Zweck habe ich ein Machine-Learning-Werkzeug entwickelt, das mittels Bloch-Simulation RF-Pulse erzeugt, basierend auf einem räumlichen Schichtprofil mit beliebiger Anzahl von Schichten und einer Menge von Randbedingungen. In Experimenten zeigen sich solche Mehrschicht-Pulse in ihrer Datenqualität vergleichbar mit zwei Einzelschicht-Pulsen (deren Messung doppelt so lange dauert).  
{{< figure src="/images/research/sms3.png" link="/images/research/sms3.png" >}}
Die obige Abbildung zeigt das Ergebnis meines Pulsgenerators für ein 4-Schicht-Ziel. Der erzeugte Puls ist oben links dargestellt, das zugehörig generierte Magnetfeldgradientenprofil befindet sich oben in der Mitte, und das resultierende Schicht-Profil ist unten rechts zu sehen.  
Die folgende Abbildung zeigt erste Messergebnisse mit einem 4-Schicht-RF-Puls zur Aufnahme eines Kugel-Phantoms. Die oberste Reihe zeigt die Schichten, die mit vier einzelnen Einzelschicht-Pulsen aufgenommen wurden; diese können als Referenz betrachtet werden. Die unterste Reihe zeigt das entsprechende Ergebnis unter Verwendung eines einzigen Pulses, der gleichzeitig vier Schichten anregt. Wie man sieht, stimmen die Ergebnisse gut überein.
{{< figure src="/images/research/4Slice.png" link="/images/research/4Slice.png" >}}
Die nächste Abbildung zeigt die Spektren eines Beispiel-Pixels aus der vierten Schicht der obigen Messung, zusammen mit den entsprechenden Spektren (schwarz: Single-Slice-Puls, blau: Multi-Slice-Puls).
{{< figure src="/images/research/slice4.png" link="/images/research/slice4.png" >}}

