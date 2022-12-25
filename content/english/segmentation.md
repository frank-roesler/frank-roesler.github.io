---
title: "Hobby Projects"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-06-01T16:49:58+01:00
draft: false
video: "english"
---

Image Segmentation with FCNs
----------------------------

My [GitHub repo](https://github.com/frank-roesler/Image_Segmentation) now contains a collection of codes includes a PyTorch implementation of the fully convolutional (FCN) version of Alexnet (see https://arxiv.org/abs/1411.4038), which detects objects in images. The code in this package has been written with one goal in mind: keeping it simple. The model is written only for 1 type of object to be detected (in addition to background), and all functions and classes are explicit; nothing is pre-trained. The goal is not to provide a powerful state-of-the-art model, but to make the basics clear.

This code was tested with the [Penn-Fudan pedestrian dataset](https://www.seas.upenn.edu/~jshi/ped_html/), as well as cat images from the [COCO dataset](https://cocodataset.org/). Even though the dataset is small, the results are reasonable. More accurate results can be obtained by using more elaborate neural network architechtures (e.g. transfer learning with a pre-trained ResNet-50).

{{< figure src="/images/seg2.png" class="center">}}
{{< figure src="/images/seg3.png" class="center">}}
{{< figure src="/images/seg4.png" class="center">}}  
(Images taken from the COCO dataset: https://cocodataset.org/)
