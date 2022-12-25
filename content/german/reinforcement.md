---
title: "Hobbyprojekte"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "german"
---

Reinforcement Learning
----------------------
Reinforcement Learning ist eine Methode, Computern beizubringen, Kontrollaufgaben zu lösen. Neuronale Netzwerke sind hier in modernen Anwengungen ein wichtiger Baustein. Hier stelle ich einen Algorithmus namend "Deep Q-Network" (DQN) vor, den ich implementiert habe, um das klassische TicTacToe (X gegen O) Spiel zu spielen.  
Kurz gesagt funktioniert ein DQN-Algorithmus, indem er jedem Zustand des Spielfeldes (d.h. jeder Anordnung von X und O) und jedem möglichen Spielzug einen *Nutzen* zuordnet. Während der Algorithmus trainiert wird, lernt das neuronal Netz, den Nutzen jedes Spielzuges in jedem Zustand vorherzusagen (Spielzüge mit hoher Gewinnwahrscheinlichkeit erhalten etwa eien großen Nutzen).  
Dies funktioniert, indem der Algorithmus für erfolgreiche Züge *belohnt* wird. Nach vielen simulierten Spielen, lernt der Algorithmus, die Belohnung (engl. *Reward*) vorherzusagen.

Das TicTacToe-Spiel, der DQN-Algorithmus, der Trainings-Loop und ein Validierungs-Skript sind in *PyTorch* implementiert und frei verfügbar auf https://github.com/frank-roesler/TicTacToe_DQN.

{{< figure src="/images/ttt.png" class="center" caption="Beispieldiagramm einiger Statistiken während des Trainingsprozesses." link="https://frank-roesler.github.io/images/ttt.png">}}
