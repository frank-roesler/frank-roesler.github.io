---
title: "Hobby Projects"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "english"
---

Denoising Signals
-----------------

Methods from Machine Learning can be used in signal processing to improve the quality of noisy signals. A toy example that I have coded in PyTorch suggests that this method works well in situations where the signal has the form of [Lorentz peaks](https://de.wikipedia.org/wiki/Lorentzkurve). Such signals are commonly found in spectroscopy data. Qualitatively, such signals look like the following image (darker blue or red = real part, brighter blue or red = imaginary part of the signal). In real life, any measured signal will not be given by a clean, smooth line, but by a noisy one, in which the individual peaks can often hardly be seen with the naked eye.

{{< figure src="/images/denoising/gt.png" link="https://frank-roesler.github.io/images/denoising/gt.png" >}}

Synthetically generated data like the above can be used to train a neural network: the network receives the noisy signal and is trained to reproduce the clean original (aka "ground truth"). The training architecture is illustrated in the figure below.

{{< figure src="/images/denoising/cnn.png" link="https://frank-roesler.github.io/images/denoising/cnn.png" >}}

My code is openly available as a Colab Notebook here: [`my_notebook`](https://colab.research.google.com/drive/1wEljoKaEuV4WR-MtUWSFjBejXchJ5oJh?usp=sharing). Anyone can download the code and (with a Google account) run it in the cloud and test it. The plots below show the result of a convolutional neural net with 6 layers trained to minimize the mean square error of its output and the clean signal. Clearly, the neural net reconstructs the original signal well, as can be seen by plotting the difference of the neural net output and the ground truth (bottom right).

{{< figure src="/images/denoising/comparison.png" link="https://frank-roesler.github.io/images/denoising/comparison.png" >}}

This project is part of an ongoing collaboration (more details here on our recent [abstract](https://www.ismrm.org/index.php?gf-download=2022%2F08%2FISMRM-Diff_Workshop_2022_ADoering.pdf&form-id=1341&field-id=16&hash=66281781958d5482202467a2c3b303a1a7d9c32327990656deee3fa54f9af27b)). A more up-to-date version of our Python code is available on the project's GitHub page: https://github.com/frank-roesler/MRspecNET (this is still work in progress).