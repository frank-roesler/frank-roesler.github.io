---
title: "Hobby Projects"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "english"
---

Teaching a Computer to Hallucinate
----------------------------------

A so-called *Generative adversarial Network (GAN)* is a combination of two neural networks (one called "generator", the other called "discriminator") designed to create images belonging to a predefined category. The two networks are set up to compete against one another: first, the generator creates an image from random noise, then the discriminator compares the generated image to a fixed training set.  
The discriminator is trained to distinguish the real images (i.e. the training set) from the forgeries, while the generator is trained to fool the discriminator. This joint training process forces the generator to produce images that are increasingly similar to the training images.  
This method was introduced in the article [[Radford, Metz, & Chintala (2015)]](https://arxiv.org/pdf/1511.06434.pdf). Inspired by [this PyTorch implementation](https://github.com/pytorch/examples/tree/main/dcgan), I tried training a GAN to produce images of cats. The results are beautiful images reminiscent of impressionist painitngs (that are only slighty creepy).

{{< figure src="/images/gan_cats.png" link="https://frank-roesler.github.io/images/gan_cats.png">}}


