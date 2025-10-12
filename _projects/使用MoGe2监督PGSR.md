---
layout: single
title: "使用MoGe2来监督PGSR"
permalink: /projects/project-2/
date: 2025-08-21
excerpt: "这使用MoGe2的单目深度估计功能渲染出normal，并以此来监督PGSR中的normal loss"
# header:
#   teaser: "/images/project-teaser-2.png"
categories:
  - Projects
---

## 项目概述
MoGe2是一个单目深度估计网络，目前属于这个领域的SOTA方法。MoGe2可以渲染出一幅图像的深度信息以及平面法向量信息，而PGSR中的loss计算中也有这部分的内容，这启示我们可以使用MoGe2来监督PGSR的normal估计。

## 技术栈
- Python
- MoGe2
- PGSR


## 项目链接
[项目分析](https://eating-cpp.github.io/posts/2025/08/blog-post-4/)
[项目实现](https://github.com/eating-cpp/Use-MoGe2-to-supervise-PGSR/tree/main)