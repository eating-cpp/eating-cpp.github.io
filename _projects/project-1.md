---
layout: single
title: "慧眼见真(OmniVerify-Nexus)——基于Omniglue的钧瓷真伪鉴别系统"
permalink: /projects/project-1/
date: 2025-05-16
excerpt: "使用OmniGlue模型实现了一个针对钧瓷真伪鉴别系统。"
header:
  teaser: "images/OmniVerify-Nexus_logo.png"
categories:
  - Projects
  - OmniGlue
---

We hereby declare that our project is developed using the publicly available model Omniglue from a 2024 CVPR paper.

![og_diagram.png](images/Omniglue_pipline.png "og_diagram.png"){: .align-center}
![OmniVerify-Nexus.png](images/OmniVerify-Nexus_logo.png "OmniVerify-Nexus.png"){: .align-center}

## Technology stack
- OmniGlue
- PyQt5

## Introduciton
Aiming at the counterfeiting problem in the Jun porcelain industry of Shenhou Town, Yuzhou, Henan, which causes annual losses exceeding 20 million yuan, our team has developed an intelligent anti-counterfeiting system for Jun porcelain based on the OmniGlue algorithm (2024 CVPR). Traditional manual identification relies on microscopic features such as glaze crazing patterns and "earthworm walking mud" patterns, suffering from defects like low efficiency and strong subjectivity. This system innovatively decouples the global shape of porcelain (using DINOv2 for semantic modeling) from local microscopic textures (Superpoint feature matching). Combined with a dynamic weight scoring mechanism, it achieves adaptive fusion of multi-angle captured images. The measured identification accuracy reaches 98.2%, and the processing time for a single image is <0.5 seconds. It supports museums, auction houses, and manufacturers in establishing permanent digital fingerprint libraries for genuine products, offering two modes: fast screening with adjustable confidence and high-precision identification. This provides an AI guardianship solution of "difficult to replicate, traceable" for the brand value of Jun porcelain, pushing the digital protection of cultural heritage into the millisecond era.

## Project URL
[More details? Check this out!](https://github.com/eating-cpp/OmniVerify-Nexus)