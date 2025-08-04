---
title: "Research"
featured_image: '/images/banner_blackboard2.jpg'
omit_header_text: true
date: 2020-05-26T11:53:04+01:00
draft: false
research_menu: "english"
---

Generating new RF pulses for MR Spectroscopy
--------------------------------------------

This is ongoing work. The code repository in its current form is available here: https://github.com/frank-roesler/Inverse_Bloch

### Overview:
Simultaneous Multi-Slice (SMS) is an established technique in magnetic resonance imaging, which enables the experimenter to drastically reduce scanning time by simultaneously measuring image data from more than one slice in the subject's body. Using SMS necessitates the parallel excitation of several layers of tissue by the same radiofrequency pulse (I have explained the basics of this [here](/bloch)).

In the related field of [Magnetic Resonance Spectroscopy](https://en.wikipedia.org/wiki/In_vivo_magnetic_resonance_spectroscopy), SMS is much less established and new pulse sequences are still at a research state. I am involved in a research project together with [André Döring](https://www.epfl.ch/labs/metmrs/research/members/andre-doring-phd/) and [Lijing Xin](https://people.epfl.ch/lijing.xin?lang=en) at EPFL Lausanne, which aims to utilise SMS to improve acquisition time in Magnetic Resonance Spectroscopic Imaging. The generation of multi slice RF pulses in spectroscopy is considerably more delicate than in classical imaging and requires a gentle balancing of diverse constraints. 

To this end, I have contributed a machine learning tool that uses Bloch simulation in order to create RF pulses given a spatial slice profile with an arbitrary number of slices and a set of constraints. In experiments, such multislice pulses are found to yield comparable data quality to two single slice pulses (whose measurement takes twice as long).  
{{< figure src="/images/research/sms3.png" link="/images/research/sms3.png" >}}
The figure above shows the result of my pulse generation tool for a 4-slice target. The resulting pulse is in the top left, the field gradient (which is also generated alongside) in the top center and the resulting slice profile is in the bottom right plot.  
The figure below shows some preliminary measurement results using a 4-slice RF pulse to scan a spherical phantom. The top row shows the slices acquired with four individual single slice pulses, which can be regarded as a ground truth to be reproduced. The bottom row shows the corresponding result using a single pulse that excites four slices at the same time. As one can see they are in good agreement.
{{< figure src="/images/research/4Slice.png" link="/images/research/4Slice.png" >}}
