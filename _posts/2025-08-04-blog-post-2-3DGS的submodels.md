---
title: '对于3DGS中的submodel的理解与实验'
date: 2025-08-04
permalink: /posts/2025/08/blog-post-2/
tags:
  - 3DGS
  - Python
  - CUDA
  - C++
---

对于3DGS中submodels的一些理解和简单实验

# 对于3DGS中的submodel的理解与实验
3DGS中的submodel指的是在3DGS中使用的子模型，例如使用的是PointNet还是PointTransformer等。她的目录结构如下所示：
```bash
submodels
|---diff-gaussian-rasterization
|   |---build
|   |---cuda_rasterizer
|   |   |---backward.cu
|   |   |---forward.cu
|   |   |---rasterizer.cu
|   |   |---rasterize_iml.cu
|   |---diff_gaussian_rasterization
|   |---diff_gaussian_rasterization.egg-info
|   |---thirdparty
|   |---ext.cpp
|   |---tasterize_points.cu
|   |---setup.py
|---fused-ssim
|---simple-knn
```
## diff-gaussian-rasterization/setup.py
将CUDA 和 C++ 编写的高斯光栅化核心代码编译为 Python 可导入的扩展模块，使得开发者可以在 Python 环境中（尤其是结合 PyTorch）调用这些高性能的底层计算功能。

1. 导入依赖库
- 从 setuptools 导入 setup 函数，用于 Python 包的配置和分发。
- 从 torch.utils.cpp_extension 导入 CUDAExtension 和 BuildExtension，用于编译和构建包含 CUDA 代码的 PyTorch 扩展模块。

2. 路径处理
- 通过 os.path 相关函数获取当前文件的绝对路径目录（代码中未直接使用，但常用于处理相对路径依赖）。

3. 包配置（setup 函数）
- `name`：指定包的名称为 `diff_gaussian_rasterization`。
- `packages`：声明包含的 Python 包列表（此处为 `diff_gaussian_rasterization`）。
- `ext_modules`：定义需要编译的扩展模块，这里通过 `CUDAExtension` 配置了 CUDA 相关的扩展：
    - `name`：扩展模块的内部名称（`diff_gaussian_rasterization._C`），供 Python 导入使用。
    - `sources`：列出需要编译的源文件，包括 CUDA 实现（`.cu` 文件）和 C++ 接口代码（`.cpp` 文件），涵盖光栅化器的核心逻辑、前向 / 反向传播等功能。
        ```cpp
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",  # CUDA光栅化器实现
                "cuda_rasterizer/forward.cu",          # 前向传播CUDA代码
                "cuda_rasterizer/backward.cu",         # 反向传播CUDA代码
                "rasterize_points.cu",                 # 点光栅化核心CUDA代码
                "ext.cpp"                              # C++扩展接口代码
                ]    
        ```
    - `extra_compile_args`：为 nvcc 编译器添加额外参数，此处指定了 `third_party/glm/` 路径作为头文件目录（`GLM` 是一个数学库，用于 CUDA 代码中的向量 / 矩阵运算）。
- `cmdclass`：指定构建扩展时使用的命令类（`BuildExtension`），用于处理 CUDA 扩展的构建流程。

## diff-gaussian-rasterization/rrasterize_points.cu
这个脚本是3DGS实时渲染算法的核心CUDA光栅化实现，主要功能是将三维空间中的3DGS分布映射到二维屏幕空间中，同时提供反向传播接口用于优化高斯分布的参数。**核心：实现了辐射场的实时渲染**。

### 主演函数与流程了解

#### 1. `RasterizeGaussiansCUDA`（正向渲染）

主要功能是:调用CUDA光栅化器`CudaRasterizer::Rasterizer::forward`，将3D高斯分布投影到2D图像,计算每个像素的颜色，融合可见告诉贡献，输出实时渲染结果。

**Input**：

- 3D高斯的核心参数：`means3D`3D位置、`colors`颜色、`opacity`不透明度、 `scales`尺度、`rotations`旋转、`cov3D_precomp`预计算的 3D 协方差矩阵。

- 相机参数：`viewmatrix`视图矩阵、`projmatrix`投影矩阵、`tan_fovx/tan_fovy`视场角正切值、`campos`相机位置。

- 渲染配置：图像尺寸`image_height/image_width`、抗锯齿`antialiasing`、是否调试`debug`等。

**Output**：

渲染的图像颜色`out_color`、每个高斯的覆盖半径`radii`、深度信息`out_invdepth`，以及 CUDA 渲染过程中使用的缓冲区`geomBuffer/binningBuffer/imgBuffer`。


#### 2. `RasterizeGaussiansBackwardCUDA`（反向传播）

计算渲染的结果对于3D高斯参数的梯度，调用CUDA光栅化器的反向接口`CudaRasterizer::Rasterizer::backward`，通过链式法则计算损失对每个3D高斯的参数的梯度，支持端到端的训练。

**Input**：
- 正向渲染的输入参数（`means3D/scales`）
- 损失函数对输出的梯度（`dL_dout_color`:颜色损失梯度，`dL_dout_invdepth`:深度损失梯度）
- 正向渲染中记录的缓冲区（`geomBuffer/binningBuffer/imgBuffer`）和覆盖半径(`radii`)

**Output**：
- 各参数的梯度：`dL_dmeans3D`（3D 位置梯度）、`dL_dcolors`（颜色梯度）、`dL_dopacity`（不透明度梯度）、`dL_dscales`（尺度梯度）、`dL_drotations`（旋转梯度）等。

#### 3. 可见性判断`markVisible`
判断3D高斯是否在相机视锥内,用于过滤不可见的高斯以提高效率.

**Input**:
- 3D 位置（`means3D`）、相机视图矩阵（`viewmatrix`）和投影矩阵（`projmatrix`）。

**Output**:
- 布尔张量(`present`), 用来标记每个高斯是否可见.


------