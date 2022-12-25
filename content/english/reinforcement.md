---
title: "Hobby Projects"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "english"
---

Reinforcement Learning
----------------------

Reinforcement Learning is a way to teach computers solve control tasks. Deep neural networks provide a powerful tool to build reinforcement learning models. This section presents an algorithm called "Deeo Q-Network" (DQN), which I have implemented to learn to play the classical TicTacToe (X vs O) game.  
In a nutshell, a DQN algorithm assigns a value to any possible move in a given state of the game (i.e. for a given configuration of X's and O's on the board, how valuable will it be to place a X in any given free space). During training, the neural net learns to predict high values for moves that are likely to lead to victory and low values for moves that are not.  
The learning process involves playing a large number of simulated games. During these games, any move that turns out to be successful (i.e. leads to victory) entails a *reward* for the algorithm. After a large number of simulated games, the algorithm learns to predict the expected reward of any given game move.

The TicTacToe game, the DQN model, the training loop and a test script that evaluates the trained model are written in PyTorch and are openly available at https://github.com/frank-roesler/TicTacToe_DQN

{{< figure src="/images/ttt.png" class="center" caption="Example training outcome for the DQN agent" link="https://frank-roesler.github.io/images/ttt.png">}}
