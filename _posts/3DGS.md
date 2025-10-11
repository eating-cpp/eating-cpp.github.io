---
title: '手撕3DGS核心，理解函数调用，为修改做准备'
date: 2025-8-12
permalink: /posts/2025/10/blog-post-1/
tags:
  - 3DGS
  - CUDA
  - C++
  - Python
---

<!-- This is a sample blog post. Lorem ipsum I can't remember the rest of lorem ipsum and don't have an internet connection right now. Testing testing testing this blog post. Blog posts are cool.

Headings are cool
======

You can have many headings
====== -->

## 1. `./scene/gaussian_model.py`的基本流程，功能和调用关系
`gaussian_model.py`文件实现了一个用于场景表示和渲染的高斯分布模型，主要功能是管理三维高斯分布的参数（位置、特征、缩放、旋转、不透明度等），并支持训练过程中的优化、剪枝、稠密化等操作。

### 1.1 核心功能概述
核心目标是通过优化这些参数，使得高斯分布在渲染时能够匹配输入图像，实现高质量的场景重建和新视角合成。

每个高斯分布包括：
- 空间位置`_xyz`
- 外观特征`_features_dc`和`_features_rest`
- 缩放参数`_scaling`
- 旋转参数`_rotation`
- 不透明度参数`_opacity`
- 二维最大半径`max_radii2D`
- 位置梯度累积`xyz_gradient_accum`
- 梯度累积的分母`denom`
- 优化器`optimizer`
- 密集点的百分比`percent_dense`
- 空间学习率缩放因子`spatial_lr_scale`

### 1.2 类与核心参数设置
#### 1.2.1 主要类：`GaussianModel`
该类封装了所有高斯分布的参数以及操作。

#### 1.2.2 激活函数
从缩放和旋转构建协方差矩阵。
<details>
<summary>代码</summary>

```python
def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            从缩放和旋转构建协方差矩阵。
            :param scaling: 缩放参数
            :param scaling_modifier: 缩放修改器
            :param rotation: 旋转参数
            :return: 对称协方差矩阵
            """
            # 构建缩放旋转矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 计算实际的协方差矩阵
            actual_covariance = L @ L.transpose(1, 2)
            # 提取对称部分
            symm = strip_symmetric(actual_covariance)
            return symm
```
</details>

通过`setup_function`初始化，选用不同的激活函数，如`torch.exp`、`torch.sigmoid`等。
<details>
<summary>代码</summary>

```py
        # 缩放激活函数，使用指数函数
        self.scaling_activation = torch.exp
        
        # 缩放逆激活函数，使用对数函数
        self.scaling_inverse_activation = torch.log

        # 协方差激活函数，使用自定义的构建函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 不透明度激活函数，使用 sigmoid 函数
        self.opacity_activation = torch.sigmoid
        
        # 不透明度逆激活函数
        self.inverse_opacity_activation = inverse_sigmoid

        # 旋转激活函数，使用归一化函数
        self.rotation_activation = torch.nn.functional.normalize
```

</details>


### 1.3 核心函数解析
#### 1.3.1 初始化与设置
- `_init_`：初始化模型参数，例如球谐阶数、优化器类型等，调用`setup_function`设置激活函数。
- `create_from_pcd`：从点云初始化模型。将点云的位置、颜色作为初始值，缩放通过点云领域距离计算，旋转初始化为单位四元数，不透明度初始化为0.1。
- `training_setup`：配置优化器（Adam或者SparseGaussianAdam）和学习率调整器，初始化梯度累计变量。

#### 1.3.2 参数访问与计算
- 以`get_*`开头的属性返回经过激活函数处理后的参数。
- `get_covariance`：通过缩放和旋转计算高斯分布的协方差矩阵，用于渲染时权重计算。

#### 1.3.3 训练过程管理
- `update_learning_rate`：根据当前迭代次数更新学习率。
- `add_densification_stats`：计算并更新密集化统计信息，用于确定是否需要进行密集化操作。

#### 1.3.4 密度调整
这是3DGS的核心创新点，通过动态增加或删除高斯点提升表示能力。
- `densify_and_clone`：对梯度大企鹅缩放小的点进行克隆（增加密度）。
- `densify_and_spilt`：对梯度大且缩放大的点进行分裂（将一个点分裂为 N 个，细化表示）。
- `densify_and_prune`：综合调用克隆、分裂，并修剪不透明度低或过大的点。
- `prune_points`：根据掩码删除不需要的点。


### 1.4 函数调用关系
#### 1.4.1 初始化流程
`__init__ → setup_functions（设置激活函数）`进行各个参数初始化后执行:
```python
# 设置激活函数和协方差构建函数
        self.setup_functions()
```

<details>
<summary>代码</summary>

```py
def setup_functions(self):
        """
        设置模型中使用的激活函数和协方差构建函数。
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """
            从缩放和旋转构建协方差矩阵。
            :param scaling: 缩放参数
            :param scaling_modifier: 缩放修改器
            :param rotation: 旋转参数
            :return: 对称协方差矩阵
            """
            # 构建缩放旋转矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 计算实际的协方差矩阵
            actual_covariance = L @ L.transpose(1, 2)
            # 提取对称部分
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 缩放激活函数，使用指数函数
        self.scaling_activation = torch.exp
        
        # 缩放逆激活函数，使用对数函数
        self.scaling_inverse_activation = torch.log

        # 协方差激活函数，使用自定义的构建函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 不透明度激活函数，使用 sigmoid 函数
        self.opacity_activation = torch.sigmoid
        
        # 不透明度逆激活函数
        self.inverse_opacity_activation = inverse_sigmoid

        # 旋转激活函数，使用归一化函数
        self.rotation_activation = torch.nn.functional.normalize
        
```

</details>

#### 1.4.2 训练循环
每一次迭代，都会有以下过程：

```py
update_learning_rate（更新学习率）→ 计算损失(在train.py中) → 反向传播(loss.backward())
add_densification_stats（累积梯度）→ 定期调用 densify_and_prune（调整点数量）
```

其中，backward实现在`submodules\diff-gaussian-rasterization\cuda_rasterizer\backward.cu`中，由s`setup.py, ext.cpp`共同封装成pytorch函数，可以自动触发。

#### 1.4.3.Densification流程：
```python
densify_and_prune → 调用 densify_and_clone 和 densify_and_split → 
densification_postfix（合并新点到模型）→ prune_points（修剪无效点）
```

## 2. `./gausasian_renderer/_init_.py`的基本流程，功能和调用关系
这个文件是一个用于渲染3D高斯点云场景的python模块，主要基于pytorch实现，同时也包含了很多submodels中的算法和函数。

**核心**：
- 定义了`render`函数,将3D场景渲染成2D图像，并返回结果及相关辅助信息。
- 依赖`torch`，用于张量计算和自动求导。
- 依赖`diff_gaussian_rasterization`，来自submodels，提供了高斯光栅化相关的配置类和光栅化器（`GaussianRasterizer`）。

### 2.1 核心流程

`train.py`中的渲染函数为`render`，定义在`./gausasian_renderer/_init_.py`中。

#### 2.1.1 初始化梯度计算张量。创建`screenspace_points`用于跟踪2D屏幕空间均值梯度。
```python
    # 创建零张量，用于让pytorch返回2D(屏幕空间)均值的梯度
    # 这里通过与pc.get_xyz相同的 dtype 和设备创建，并设置requires_grad=True以启用梯度计算
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 保留梯度，以便后续可以访问梯度信息
        screenspace_points.retain_grad()
    except:
        pass
```
这里面，`pc`是一个`GaussianModel`的对象实例。

#### 2.1.2 配置光栅化参数

- 基于相机视场角计算水平、垂直方向的正切值
```python
# 计算水平和垂直方向的视场角正切值（FoV的一半）
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
```

- 初始化光栅器对象，使用到来自`submodels`中的`GaussianRasterizationSettings`和`GaussianRasterizer`类。
```python
 # 初始化光栅化设置对象
    raster_settings = GaussianRasterizationSettings(# 来自submodels
        image_height=int(viewpoint_camera.image_height),  # 图像高度
        image_width=int(viewpoint_camera.image_width),    # 图像宽度
        tanfovx=tanfovx,                                  # 水平视场角正切值
        tanfovy=tanfovy,                                  # 垂直视场角正切值
        bg=bg_color,                                      # 背景颜色
        scale_modifier=scaling_modifier,                  # 缩放修正因子
        viewmatrix=viewpoint_camera.world_view_transform, # 视图矩阵（世界到相机坐标系转换）
        projmatrix=viewpoint_camera.full_proj_transform,  # 投影矩阵（相机坐标到裁剪坐标转换）
        sh_degree=pc.active_sh_degree,                    # 球谐函数阶数
        campos=viewpoint_camera.camera_center,            # 相机位置（世界坐标系）
        prefiltered=False,                                # 是否预过滤（默认关闭）
        debug=pipe.debug,                                 # 是否开启调试模式
        antialiasing=pipe.antialiasing                    # 是否开启抗锯齿
    )

    # 创建光栅化器实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)# 来自submodels
```

其中，`GaussianRasterizationSettings`是一个配置类，用于存储光栅化参数，`GaussianRasterizer`是一个光栅化器类，用于将3D高斯点云渲染成2D图像。
他们都声明在`gaussian_renderer\__init__.py`中。该文件稍后会讲到


#### 2.1.3 准备高斯点云数据
- 获取高斯点云的3D坐标
```python
# 获取高斯点云的3D坐标
    means3D = pc.get_xyz
```
`get_xyz`是`GaussianModel`类中的一个方法，用于获取所有高斯点的位置坐标。

- 获取2D，即屏幕空间点的坐标
```python
# 获取屏幕空间点的坐标
    means2D = screenspace_points
```
这个主要是用于计算高斯点在屏幕空间的位置，用于后续的光栅化。

- 获取高斯点的不透明度
```python
# 获取高斯点的不透明度
    opacity = pc.get_opacity
```

- 获取3D协方差
```python
# 如果提供了预计算的3D协方差，则使用它；否则由光栅化器从缩放/旋转计算
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        # 在Python中计算3D协方差
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 从高斯模型获取缩放和旋转参数（由光栅化器计算协方差）
        scales = pc.get_scaling
        rotations = pc.get_rotation
```
`get_covariance`是`GaussianModel`类中的一个方法，用于获取所有高斯点的3D协方差矩阵。

D 协方差矩阵（3D Covariance Matrix）主要用于描述高斯分布在 3D 空间中的形状、大小和朝向，其核心作用是定义单个高斯点在 3D 空间中的空间分布特征。

#### 2.1.4 处理颜色信息
- 如果提供了覆盖颜色(`override_color`)，则直接使用，否则通过球谐函数来计算颜色：可以在python中使用`eval_sh`转换，或者交由光栅化器计算。
```python
 # 如果提供了预计算的颜色，则使用它们；否则根据配置决定在Python中还是光栅化器中计算球谐函数到RGB的转换
    shs = None
    colors_precomp = None
    if override_color is None:
        # 未提供覆盖颜色，使用高斯模型自身的特征计算
        if pipe.convert_SHs_python:
            # 在Python中转换球谐函数到RGB
            # 调整特征张量形状以适应球谐函数评估
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 计算高斯点到相机中心的方向向量
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # 归一化方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 评估球谐函数得到RGB颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 调整颜色范围并限制最小值为0
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 不在Python中转换，由光栅化器处理
            if separate_sh:
                # 分离球谐函数的直流分量(DC)和其余分量
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                # 使用完整的球谐函数特征
                shs = pc.get_features
    else:
        # 使用提供的覆盖颜色
        colors_precomp = override_color
```
`get_features`是`GaussianModel`类中的一个方法，用于获取所有高斯点的颜色特征。

`pipe`是`PipelineParams`类的一个实例，用于存储和管理渲染管道的参数。在`trian.py`中有如下定义：`pp = PipelineParams(parser)# 类：PipelineParams 来自arguments/_init_.py`，随后传入`trian`方法：
```python
# 开始训练 来自./train.py
    training(
        lp.extract(args),
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from
        )
    #training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)
```

#### 2.1.5 执行光栅化
- 调用`rasterizer`渲染可见的高斯点，得到图像、半径、深度等信息。
```python
# 光栅化可见的高斯点到图像，并获取它们在屏幕上的半径
    if separate_sh:
        # 分离球谐函数分量的情况
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,          # 3D坐标
            means2D = means2D,          # 2D屏幕坐标（用于梯度）
            dc = dc,                    # 球谐函数直流分量
            shs = shs,                  # 球谐函数其余分量
            colors_precomp = colors_precomp,  # 预计算的颜色（如果有）
            opacities = opacity,        # 不透明度
            scales = scales,            # 缩放参数
            rotations = rotations,      # 旋转参数
            cov3D_precomp = cov3D_precomp)  # 预计算的3D协方差（如果有）
    else:
        # 不分离球谐函数分量的情况
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
```
`rasterizer`是`GaussianRasterizer`类的一个实例，用于执行高斯点的光栅化渲染，在`submodels`中实现。

#### 2.1.6 后处理
- 如果启用了训练曝光，那么应用训练曝光,并将渲染图形的像素值限制在[0,1]范围内

```python
 # 对渲染图像应用曝光（仅训练时使用）
    if use_trained_exp:
        # 根据相机图像名称获取对应的曝光参数
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        # 应用曝光变换（矩阵乘法 + 偏移）
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    rendered_image = rendered_image.clamp(0, 1)
```
`clamp`函数是 PyTorch 中的一个张量操作函数（`torch.clamp`），其核心功能是将张量中的元素值限制在指定的范围内，超出范围的值会被截断到边界值。

#### 2.1.7 返回结果
- 构建输出字典
```python
# 构建输出字典
    out = {
        "render": rendered_image,                  # 渲染得到的图像
        "viewspace_points": screenspace_points,    # 视图空间点（用于梯度）
        "visibility_filter": (radii > 0).nonzero(), # 可见性过滤器（半径>0的点）
        "radii": radii,                            # 高斯点在屏幕上的半径
        "depth": depth_image                       # 深度图像
    }
    
    return out
```
该代码是 3D 高斯点云渲染技术（如 Gaussian Splatting）中的核心组件，负责将 3D 高斯表示转换为可视化的 2D 图像。

## 3. `submodules\diff-gaussian-rasterization\diff_gaussian_rasterization\__init__.py`流程与调用理解
该文件会经由`setup.py`安装到`site-packages`中，`setup.py`中定义了安装的内容，包括`diff_gaussian_rasterization`包的名称、版本、作者、描述、依赖项、入口点等。

```py
# 导入setuptools库，用于Python包的构建和分发
from setuptools import setup
# 导入PyTorch的CUDA扩展工具，用于编译CUDA相关代码
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# 获取当前文件所在目录的绝对路径（虽然这里未直接使用，但通常用于处理相对路径）
os.path.dirname(os.path.abspath(__file__))

# 配置并构建Python扩展包
setup(
    # 包的名称：diff_gaussian_rasterization
    name="diff_gaussian_rasterization",
    # 指定包含的Python包列表
    packages=['diff_gaussian_rasterization'],
    # 定义扩展模块（包含CUDA代码）
    ext_modules=[
        CUDAExtension(
            # 扩展模块的内部名称（供Python导入使用）
            name="diff_gaussian_rasterization._C",
            # 需要编译的源文件列表（包含CUDA实现和C++绑定代码）
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",  # CUDA光栅化器实现
                "cuda_rasterizer/forward.cu",          # 前向传播CUDA代码
                "cuda_rasterizer/backward.cu",         # 反向传播CUDA代码
                "rasterize_points.cu",                 # 点光栅化核心CUDA代码
                "ext.cpp"                              # C++扩展接口代码
            ],
            # 额外的编译参数（针对nvcc编译器）
            # 这里添加了glm库的路径，用于CUDA代码中的数学运算
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
        )
    ],
    # 指定构建扩展时使用的命令类
    # BuildExtension用于处理CUDA扩展的构建过程
    cmdclass={
        'build_ext': BuildExtension
    }
)
```
会将导入的包命名为`_C`，这是一个C++扩展模块，用于实现CUDA代码的调用。


真正使用的路径为：`E:\progrrams\Anaconda\myenvs\gaussian_splatting\Lib\site-packages\diff_gaussian_rasterization\__init__.py`

### 3.1 核心功能
这个代码主要实现了将3D高斯分布渲染到2D图像平面，包括钱箱渲染计算和反向传播计算。

### 3.2 主要组件

#### 3.2.1 辅助函数
- `cpu_deep_copy_tuple`：深度复制元组中的元素，将其中的PyTorch张量复制到CPU上。
```python
def cpu_deep_copy_tuple(input_tuple):
    """
    深度复制元组中的元素，将其中的PyTorch张量复制到CPU上
    
    参数:
        input_tuple: 包含可能包含PyTorch张量的元组
    返回:
        新元组，其中的张量已被复制到CPU并克隆，非张量元素保持不变
    """
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)
```

#### 3.2.2 光栅化核心函数
- `rasterize_gaussians`：高斯光栅化的封装函数，调用自定义的求导函数`_RasterizeGaussians`实现核心逻辑。
```python
def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    """
    高斯光栅化的封装函数，调用自定义的自动求导函数_RasterizeGaussians
    
    参数:
        means3D: 高斯分布的3D中心点坐标
        means2D: 高斯分布的2D投影中心点坐标（可选）
        sh: 球谐函数系数，用于计算颜色
        colors_precomp: 预计算的颜色值（与sh二选一）
        opacities: 高斯分布的不透明度
        scales: 高斯分布的缩放因子（与cov3Ds_precomp二选一）
        rotations: 高斯分布的旋转参数（与cov3Ds_precomp二选一）
        cov3Ds_precomp: 预计算的3D协方差矩阵（与scales和rotations二选一）
        raster_settings: 光栅化设置参数对象
    返回:
        渲染结果（颜色、半径、逆深度）
    """
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
```
包含一个调用关系：`rasterize_gaussians → _RasterizeGaussians`。

注意到有`apply`函数，这是一个自定义的自动求导函数(`torch.autograd.Function `子类)的核心调用方式，用于触发前向传播。

具体而言，pytorch中，如果需要自定义一个支持自动微分的操作，（即同时实现前向传播和反向传播），需要继承`torch.autograd.Function`类，并重写`forward`和`backward`方法，这些类不能像普通类一样通过`_init_`实例化后调用，**必须通过`apply`方法来触发**。

自动触发下面提到的`forward`方法。

#### 3.2.3 自定义自动求导类`_RasterizeGaussians`
继承自`torch.autograd.Function`类，用于实现高斯光栅化的前向传播和反向传播。
在这其中，定义了`forward`和`backward`方法，分别实现前向传播和反向传播。

在这里导入了一个包`from . import _C  # 导入C++/CUDA扩展模块，用于高斯光栅化的底层实现`这里的`_C`是一个C++/CUDA扩展模块，用于实现高斯光栅化的底层计算。在`submodules\diff-gaussian-rasterization\setup.py`中由 `diff_gaussian_rasterization`封装CUDA和C++底层代码，并命名扩展模块名称为`_C`。

- **前向传播**
```python
class _RasterizeGaussians(torch.autograd.Function):
    """
    自定义PyTorch自动求导函数，实现高斯光栅化的前向和反向传播
    用于将3D高斯分布渲染到2D图像平面，并支持自动微分
    """
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        """
        前向传播：调用C++/CUDA后端执行高斯光栅化
        
        参数:
            ctx: 上下文对象，用于保存反向传播所需的数据
            其他参数: 同rasterize_gaussians函数
        返回:
            color: 渲染得到的颜色图像
            radii: 每个高斯在图像平面上的半径
            invdepths: 逆深度图
        """

        # 重组参数，使其符合C++库的预期格式
        args = (
            raster_settings.bg,  # 背景颜色
            means3D,             # 3D中心点
            colors_precomp,      # 预计算颜色
            opacities,           # 不透明度
            scales,              # 缩放因子
            rotations,           # 旋转参数
            raster_settings.scale_modifier,  # 缩放修正因子
            cov3Ds_precomp,      # 预计算协方差矩阵
            raster_settings.viewmatrix,      # 视图矩阵
            raster_settings.projmatrix,      # 投影矩阵
            raster_settings.tanfovx,         # x方向视场角的正切值
            raster_settings.tanfovy,         # y方向视场角的正切值
            raster_settings.image_height,    # 图像高度
            raster_settings.image_width,     # 图像宽度
            sh,                   # 球谐函数系数
            raster_settings.sh_degree,       # 球谐函数阶数
            raster_settings.campos,          # 相机位置
            raster_settings.prefiltered,     # 是否预过滤
            raster_settings.antialiasing,    # 是否启用抗锯齿
            raster_settings.debug            # 是否启用调试模式
        )

        # 调用C++/CUDA光栅化器
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # 保存反向传播所需的张量和设置
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths
```
调用`_C.rasterize_gaussians`函数，该函数是C++/CUDA扩展模块`_C`中的一个函数，用于执行高斯光栅化的计算，输出渲染的颜色图像，高斯半径，逆深度图等。

- **反向传播**
```python
 def backward(ctx, grad_out_color, _, grad_out_depth):
        """
        反向传播：计算输入参数的梯度
        
        参数:
            ctx: 上下文对象，包含前向传播保存的数据
            grad_out_color: 输出颜色的梯度
            grad_out_depth: 输出深度的梯度
        返回:
            各输入参数的梯度（与前向输入参数一一对应）
        """

        # 从上下文中恢复必要的值
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # 重组参数，使其符合C++后端的预期格式
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,    # 颜色梯度输入
                grad_out_depth,    # 深度梯度输入
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,        # 前向传播保存的几何缓冲区
                num_rendered,      # 渲染的高斯数量
                binningBuffer,     # 前向传播保存的分箱缓冲区
                imgBuffer,         # 前向传播保存的图像缓冲区
                raster_settings.antialiasing,
                raster_settings.debug)

        # 调用后端反向传播函数计算梯度
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        # 组织梯度返回值（与前向输入参数顺序一致，不需要梯度的参数返回None）
        grads = (
            grad_means3D,          # means3D的梯度
            grad_means2D,          # means2D的梯度
            grad_sh,               # sh的梯度
            grad_colors_precomp,   # colors_precomp的梯度
            grad_opacities,        # opacities的梯度
            grad_scales,           # scales的梯度
            grad_rotations,        # rotations的梯度
            grad_cov3Ds_precomp,   # cov3Ds_precomp的梯度
            None,                  # raster_settings不需要梯度
        )

        return grads
```
调用`_C.rasterize_gaussians_backward`函数，该函数是C++/CUDA扩展模块`_C`中的一个函数，用于执行高斯光栅化的反向传播计算，输出输入参数的梯度。计算输入参数的梯度，用于模型训练师的参数更新。

触发方式在上面也提到了，`train.py`中的`loss.backward()`

#### 3.2.4 光栅化配置类`GaussianRasterizationSettings`
基于`NamedTuple`的配置集合，包含渲染的所有参数。
```python
class GaussianRasterizationSettings(NamedTuple):
    """
    高斯光栅化的配置参数集合（命名元组）
    
    属性:
        image_height: 输出图像高度
        image_width: 输出图像宽度
        tanfovx: x方向视场角的正切值 (tan(fov_x / 2))
        tanfovy: y方向视场角的正切值 (tan(fov_y / 2))
        bg: 背景颜色张量 (3元素，RGB)
        scale_modifier: 全局缩放修正因子
        viewmatrix: 视图矩阵 (4x4)，将世界坐标转换到相机坐标
        projmatrix: 投影矩阵 (4x4)，将相机坐标转换到裁剪坐标
        sh_degree: 球谐函数的阶数 (0表示只使用常数项)
        campos: 相机在世界坐标系中的位置 (3元素)
        prefiltered: 是否对高斯进行预过滤（提高性能）
        debug: 是否启用调试模式（输出额外信息）
        antialiasing: 是否启用抗锯齿
    """
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
```

#### 3.2.5 光栅化类`GaussianRasterizer`
- 继承自`nn.Module`类，封装光栅化逻辑为PyTorch模块。
- `markVisible`方法：基于视锥体剔除，标记在相机视野内的 3D 高斯（不跟踪梯度，仅用于筛选可见高斯）。
```python
def markVisible(self, positions):
        """
        标记在相机视锥体内可见的3D点
        
        参数:
            positions: 3D点坐标张量 (N, 3)
        返回:
            布尔张量 (N,)，表示每个点是否可见
        """
        # 基于视锥体剔除标记可见点（不跟踪梯度）
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,  # 视图矩阵
                raster_settings.projmatrix)  # 投影矩阵
            
        return visible
```

- `forward`方法：执行前向传播，渲染 3D 高斯到 2D 图像。
```python
def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        """
        前向传播：执行高斯光栅化
        
        参数:
            means3D: 高斯3D中心点 (N, 3)
            means2D: 高斯2D投影中心点（可选，(N, 2)）
            opacities: 高斯不透明度 (N, 1)
            shs: 球谐函数系数（可选，用于计算颜色）
            colors_precomp: 预计算颜色（可选，与shs二选一）
            scales: 高斯缩放因子（可选，与cov3D_precomp二选一）
            rotations: 高斯旋转参数（可选，与cov3D_precomp二选一）
            cov3D_precomp: 预计算3D协方差矩阵（可选，与scales和rotations二选一）
        返回:
            渲染结果（颜色图像、半径、逆深度图）
        """
        
        raster_settings = self.raster_settings

        # 检查颜色输入的合法性（必须提供shs或colors_precomp中的一个）
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')
        
        # 检查协方差输入的合法性（必须提供scales/rotations对或cov3D_precomp中的一个）
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # 为未提供的可选参数设置空张量
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # 调用C++/CUDA光栅化函数
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
```
存在一个调用关系：`return rasterize_gaussians`。

这里`rasterize_gaussians`方法的名称由文件`submodules\diff-gaussian-rasterization\ext.cpp`命名，具体的实现这是在文件`submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer_impl.cu`中。

```cpp
#include <torch/extension.h>
#include <E:\personal\code\3DGS\gaussian-splatting\submodules\diff-gaussian-rasterization\rasterize_points.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
```


## 4.`submodules\diff-gaussian-rasterization\setup.py`都干了什么
它是一个构建和分发 Python 扩展包的setup.py文件，主要功能是编译包含 CUDA 和 C++ 代码的扩展模块。

具体而言：
```python
ext_modules=[
    CUDAExtension(
        name="diff_gaussian_rasterization._C",  # 扩展模块的内部名称（导入时需用此名称）
        sources=[  # 需要编译的源文件列表
            "cuda_rasterizer/rasterizer_impl.cu",  # CUDA光栅化器实现
            "cuda_rasterizer/forward.cu",          # 前向传播的CUDA代码
            "cuda_rasterizer/backward.cu",         # 反向传播的CUDA代码
            "rasterize_points.cu",                 # 点光栅化核心逻辑的CUDA代码
            "ext.cpp"                              # C++代码，用于连接CUDA与Python的接口
        ],
        # 额外编译参数（针对nvcc编译器，CUDA的编译器）
        extra_compile_args={
            "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]
        }
    )
]
```
- `name`：扩展模块的内部标志，最终会生成一个名为`_C`的扩展模块，放在`diff_gaussian_rasterization`包下，可通过`from diff_gaussian_rasterization import _C`导入。

- `source`包含了需要编译的源文件。

下面逐一分析`source`中的每个文件的核心流程与调用关系。

## 5. `submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer_impl.cu`核心功能与调用关系理解

**核心功能**：在CUDA中实现了3D高斯分布的可微分光栅化（forward渲染和backward梯度计算），支持将3D高斯投影到2D图像

**主要技术栈**：C++、CUDA（GPU并行计算）、GLM（数学库）、CUB（CUDA并行算法库，用于排序，前缀和等）。

### 5.1 辅助函数

#### 5.1.1 `getHigherMsb` CPU
- 主要实现计算整数最高有效位的下一位，用于后续排序时确定位范围
- 这段代码主要是在cpu上运行。
```C++
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}
```

#### 5.1.2 `checkFrustum` CUDA内核
- 判断3D高斯是否在相机视锥内，标记可见的高斯分布，用于裁剪不可见的高斯，提升效率。
```C++
// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(
        int P,//高斯点的数量
	const float* orig_points,//指向存储原始高斯点坐标的数组的指针
	const float* viewmatrix,//指向视图矩阵的指针，用于将点从世界坐标系转换到视图坐标系
	const float* projmatrix,//指向投影矩阵的指针，用于将点从视图坐标系转换到裁剪坐标系

	bool* present//指向布尔类型数组的指针，用于标记每个高斯点是否在视锥内
        )
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);//调用 in_frustum 函数对当前索引对应的高斯点进行视锥测试，将测试结果存储在 present 数组中。in_frustum 函数的返回值为 true 表示该点在视锥内，false 表示不在视锥内。
}
```

- `auto idx = cg::this_grid().thread_rank();`
每个线程处理一个高斯点，使用`cg::this_grid().thread_rank()`获取当前线程的全局索引`idx`，用于确定当前线程处理的高斯点的索引。


- `in_frustum`：
**视锥测试函数,定义在`submodules\diff-gaussian-rasterization\cuda_rasterizer\auxiliary.h`中**.
该头文件函数主要功能:接收点的索引、原始点坐标数组、视图矩阵、投影矩阵、预过滤标志和视图空间点引用作为参数，将点转换到屏幕空间和视图空间后，根据点在视图空间的深度值判断点是否在视锥体内。若点的深度值小于等于 0.2 且预过滤标志为真，则打印错误信息并终止程序；否则返回该点是否在视锥体内的布尔值。

### 5.2 高斯与tile的匹配
为了高效并行渲染,代码将图像划分为多个tile,并为每个tile分配需要渲染的高斯.

#### 5.2.1 `duplicateWithKeys`CUDA内核
核心功能是为每个可见的高斯点生成其与覆盖图块对应的键值对，键包含图块 ID 和深度信息，值为高斯点的索引。这些键值对后续可用于排序，使得高斯点按图块和深度有序排列。
```c++
__global__ void duplicateWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{
    // 获取当前线程在网格中的全局索引
    auto idx = cg::this_grid().thread_rank();
    // 边界检查，如果线程索引超出高斯点总数，直接返回
    if (idx >= P)
        return;

    // 仅为可见的高斯点生成键值对，半径大于 0 表示该高斯点可见
    if (radii[idx] > 0)
    {
        // 找到当前高斯点在缓冲区中写入键值对的偏移量
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;

        // 调用 getRect 函数获取当前高斯点覆盖的图块矩形范围
        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // 遍历高斯点覆盖的每个图块
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                // 计算图块 ID
                uint64_t key = y * grid.x + x;
                // 将图块 ID 左移 32 位
                key <<= 32;
                // 将深度值合并到键中
                key |= *((uint32_t*)&depths[idx]);
                // 存储键
                gaussian_keys_unsorted[off] = key;
                // 存储值，即高斯点的索引
                gaussian_values_unsorted[off] = idx;
                // 偏移量递增
                off++;
            }
        }
    }
}
```
- `cg::this_grid().thread_rank()`: 获取当前线程的全局索引`idx`
- 可见性检查: 通过判断`radii[idx] > 0`来确定该高斯点是否可见, 仅仅为可见高斯点生成键值对.
- 键值对生成: 便利高斯点覆盖的每个图块, 为每个图块生成一个键值对, **键为图块ID和深度, 值为高斯点索引**.

#### 5.2.2 `identifyTileRanges` CUDA内核
对排序后的键值对进行分析，确定每个瓦片对应的高斯索引范围（`start/end`），便于后续瓦片并行渲染。

具体而言: 函数的主要作用是检查 `point_list_keys` 数组中的键，确定每个图块（tile）在完整排序列表中的起始和结束位置，然后将这些范围信息存储到 `ranges`数组里。该函数会为每个实例化（重复）的高斯 ID 执行一次。

```cpp
// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();  // 获取当前线程在网格中的全局索引
    if (idx >= L)  // 如果索引超出列表长度，直接返回
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = point_list_keys[idx];  // 获取当前索引对应的键
    uint32_t currtile = key >> 32;  // 通过右移 32 位获取当前键对应的图块 ID
    if (idx == 0)  // 如果是第一个元素
        ranges[currtile].x = 0;  // 设置当前图块的起始位置为 0
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;  // 获取前一个元素对应的图块 ID
        if (currtile != prevtile)  // 如果当前图块 ID 与前一个不同
        {
            ranges[prevtile].y = idx;  // 设置前一个图块的结束位置为当前索引
            ranges[currtile].x = idx;  // 设置当前图块的起始位置为当前索引
        }
    }
    if (idx == L - 1)  // 如果是最后一个元素
        ranges[currtile].y = L;  // 设置当前图块的结束位置为列表长度
}

```

- `auto idx = cg::this_grid().thread_rank();`: 获取当前线程的全局索引`idx`.


- 获取当前图块的ID:
```cpp
uint64_t key = point_list_keys[idx];  // 获取当前索引对应的键
    uint32_t currtile = key >> 32;
```

### 5.3 正向渲染
`CudaRasterizer::Rasterizer::forward`是核心渲染函数, 实现了高斯客卫光栅化的前向渲染过程. 

#### 5.3.1 计算焦距
根据视场角和图像尺寸计算水平和垂直焦距
```cpp
const float focal_y = height / (2.0f * tan_fovy);
const float focal_x = width / (2.0f * tan_fovx);
```

#### 5.3.2 初始化几何状态
从 `geometryBuffer` 获取内存并初始化 `GeometryState`。
```cpp
size_t chunk_size = required<GeometryState>(P);
char* chunkptr = geometryBuffer(chunk_size);
GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
```

#### 5.3.3 定义grid和block
义用于 CUDA 并行计算的网格和线程块, 用于实现图像tile的划分,每一个tile由一个线程处理.
```cpp
dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);  // 瓦片网格（x方向瓦片数，y方向瓦片数）
dim3 block(BLOCK_X, BLOCK_Y, 1);  // 每个瓦片的线程块大小（如16x16）
```
- 例如: 若图像宽 1024、高 768，`BLOCK_X=16`、`BLOCK_Y=16`，则 `tile_grid` 为 (64, 48)，共 `64×48=3072` 个瓦片。

#### 5.3.4 高斯预处理

这是前行渲染的核心预处理步骤，在GPU上执行，对每个3D高斯完成。
```cpp
CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)
```
代码中的调用出现在`FORWARD::preprocess(···)`.
- 在`forward.h`中被声明
```cpp
namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);
    //...
}
#endif
```


- 这个函数的实现在`submodules\diff-gaussian-rasterization\cuda_rasterizer\forward.cu`这个文件中,下面谈下对他的理解

 ##### 5.3.4.1 `submodules\diff-gaussian-rasterization\cuda_rasterizer\forward.cu`中的`preprocess`理解

该函数的主要功能是对输入的高斯点数据进行预处理，包括视锥体裁剪、坐标转换、协方差矩阵计算、颜色计算等操作，为后续的光栅化过程做准备。处理完成后，会将有用的辅助数据存储起来，供后续步骤使用。

代码中存在一个这样的调用方式:`preprocess → preprocessCUDA`
```cpp
void FORWARD::preprocess(···)
{
    preprocessCUDApreprocessCUDA<NUM_CHANNELS> <<<(P + 255) / 256, 256 >>>(···)
}
```

preprocessCUDA 内核函数：
```cpp
// ... existing code ...
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    float2* points_xy_image,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered,
    bool antialiasing)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // 初始化半径和接触的图块数为 0
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // 视锥体裁剪，若在视锥体外则跳过
    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // 投影变换
    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // 计算 3D 协方差矩阵
    const float* cov3D;
    if (cov3D_precomp != nullptr) {
        cov3D = cov3D_precomp + idx * 6;
    } else {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    // 计算 2D 屏幕空间协方差矩阵
    float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // 抗锯齿处理
    constexpr float h_var = 0.3f;
    const float det_cov = cov.x * cov.z - cov.y * cov.y;
    cov.x += h_var;
    cov.z += h_var;
    const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
    float h_convolution_scaling = 1.0f;
    if(antialiasing)
        h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov));

    // 求逆协方差矩阵
    const float det = det_cov_plus_h_cov;
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    // 计算屏幕空间范围
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // 计算颜色
    if (colors_precomp == nullptr) {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // 存储辅助数据
    depths[idx] = p_view.z;
    radii[idx] = my_radius;
    points_xy_image[idx] = point_image;
    float opacity = opacities[idx];
    conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}
// ... existing code ...

```
###### 执行流程总结

1. **线程索引检查**：获取当前线程的全局索引，若索引超出高斯点数量则返回。
2. **初始化**：将当前高斯点的半径和接触的图块数初始化为 0。
3. **视锥体裁剪**：调用 in_frustum 函数检查当前高斯点是否在视锥体内，若不在则返回。
4. **投影变换**：将高斯点从世界空间转换到裁剪空间。
5. **协方差矩阵计算**：计算 3D 协方差矩阵和 2D 屏幕空间协方差矩阵。
6. **抗锯齿处理**：若开启抗锯齿，对协方差矩阵进行相应处理。
7. **求逆协方差矩阵**：使用 EWA 算法求逆协方差矩阵。
8. **计算屏幕空间范围**：计算当前高斯点在屏幕空间的覆盖范围，若覆盖范围为 0 则返回。
9. **颜色计算**：若颜色未预计算，则调用 computeColorFromSH 函数将球谐系数转换为 RGB 颜色。
10. **存储辅助数据**：将处理结果存储到相应的输出数组中

回到我们的`rasterizer_impl.cu`文件, 对每个高斯都做如下的操作:
1. 3D 到 2D 投影：
通过视图矩阵（viewmatrix）和投影矩阵（projmatrix）将 means3D（3D 中心）转换为 2D 图像坐标 geomState.means2D，并计算深度 geomState.depths。
2. 协方差矩阵计算：
若未提供预计算的 3D 协方差（cov3D_precomp），则根据 scales（尺度）和 rotations（旋转四元数）计算 3D 协方差矩阵，并投影为 2D 协方差（用于表示高斯在 2D 图像上的形状），存储于 geomState.cov3D。
3. 颜色计算：
若提供 colors_precomp（预计算颜色），直接使用；
否则通过球谐函数（shs）和相机位置（cam_pos）计算光照后的颜色，存储于 geomState.rgb。
4. 半径与视锥体裁剪：
计算高斯在 2D 图像上的覆盖半径 radii，并通过视锥体测试过滤不可见高斯（半径设为 0）。同时记录每个高斯覆盖的瓦片数量 geomState.tiles_touched。

#### 5.3.5 tile数量前缀和计算
主要是为了后续tile-gaussian匹配分配内存，通过`Prefix Sum`前缀和来计算每个高斯在缓冲区中的偏移量.
```cpp
cub::DeviceScan::InclusiveSum(
  geomState.scanning_space,  // 临时空间
  geomState.scan_size,       // 临时空间大小
  geomState.tiles_touched,   // 输入：每个高斯覆盖的瓦片数（如[2,3,0,2,1]）
  geomState.point_offsets,   // 输出：前缀和结果（如[2,5,5,7,8]）
  P                          // 高斯数量
);
```
#### 5.3.6 tile-gaussian匹配与排序
##### 5.3.6.1 生成键值对
为每个高斯覆盖的tile生成一个键值对
- key：`tile_id << 32 | depth` 高 32 位为瓦片 ID，低 32 位为深度，确保排序后高斯按 “瓦片→深度” 顺序排列（深度用于透明度混合）
- value: 高斯的索引（`idx`），用于关联具体高斯参数, 结果存储于` binningState.point_list_keys_unsorted（未排序键）`和 `binningState.point_list_unsorted（未排序值）`。

#### 5.3.6.2 键值对的排序
使用CUB库的基数排序对键值对进行排序，确保同一个tile的高斯按照深度递增排序.
```cpp
CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
    binningState.list_sorting_space,  // 临时空间
  binningState.sorting_size,        // 临时空间大小
  binningState.point_list_keys_unsorted,  // 输入键
  binningState.point_list_keys,            // 输出排序键
  binningState.point_list_unsorted,        // 输入值
  binningState.point_list,                // 输出排序值
  num_rendered, 0, 32 + bit  // 排序范围：覆盖瓦片ID（高32位）和深度（低32位）
),
debug);

```
#### 5.3.7 标记tile-gaussian范围(`identifyTileRanges`内核)
对排序后的键值对分析，确定每个瓦片对应的高斯索引范围（start 和 end），存储于 `imgState.ranges`（`uint2` 类型，x 为起始索引，y 为结束索引）。
```cpp
// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

```
例如：瓦片 k 的高斯索引范围为 `[ranges[k].x, ranges[k].y]`，便于后续瓦片并行渲染时快速定位需处理的高斯

`identifyTileRanges`的设备函数实现如下:
```cpp
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}
```


#### 5.3.8 并行渲染(`FORWARD::render`)
每个tile的线程块并行渲染器范围内的高斯.
```cpp
CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)
```
<!-- 写一个render的cuda实现,应该是个函数 -->
`FORWARD::render`定义在`submodules\diff-gaussian-rasterization\cuda_rasterizer\forward.h`中:
```cpp
namespace FORWARD
{
    //...

    void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth);
}
```
具体实现在`submodules\diff-gaussian-rasterization\cuda_rasterizer\forward.cu`中:
```cpp
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}
```



### 5.4 反向传播
`CudaRasterizer::Rasterizer::backward`用于计算渲染结果对输入参数的梯度，支持端到端优化.
```cpp
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
```
主要执行了两个关键操作，分别是反向传播的渲染（BACKWARD::render）和预处理（BACKWARD::preprocess），并且在预处理前会根据条件选择使用预计算的协方差矩阵还是自行计算的协方差矩阵。

存在两个调用`BACKWARD::render`和`BACKWARD::preprocess`, 他们均在`backward.h`中定义, 具体实现在`backward.cu`中.


#### 5.4.1 反向传播渲染(`BACKWARD::render`)
#### 5.4.2 反向传播预处理(`BACKWARD::preprocess`)



## 6. `submodules\diff-gaussian-rasterization\rasterize_points.cu`干了什么
代码实现了3D高斯分布体素的CUDA加速光栅化, 包括:
1. 正向渲染: 将3D高斯分布体素投影到2D图像, 生成渲染结果.
2. 反向传播: 计算损失函数对输入参数的梯度,支持端到端训练.
3. 可见性判断: 标记当前视角下可见的高斯点.

### 6.1 关键函数理解
#### 6.1.1 辅助函数: `resizeFunctional`
这个函数主要是用于创建一个用于动态调整PyTorch张量大小的函数对象, 供CUDA光栅化器内部使用. 
```cpp
/**
 * 创建一个用于调整Tensor大小的函数对象
 * 该函数会被CUDA光栅化器用于动态调整缓冲区大小
 * 
 * @param t 需要调整大小的PyTorch张量
 * @return 一个函数对象，接收大小参数N，调整张量大小并返回数据指针
 */
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    // 定义lambda函数，捕获张量t的引用
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});  // 调整张量大小
        // 将张量数据指针转换为char*并返回
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}
```
CUDA 光栅化过程中需要动态调整缓冲区大小（如几何缓冲区、分箱缓冲区），该函数提供了灵活的内存管理能力.
在后续的代码中, 主要是在`RasterizeGaussiansCUDA`中有调用,具体如下:
```cpp
// 创建缓冲区大小调整函数
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
```

#### 6.1.2 正向渲染函数: `RasterizeGaussiansCUDA`
核心功能: 将3D高斯分布体素光栅化到2D图像, 返回渲染结果及中间数据.
```cpp
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
{
    // 输入验证：检查means3D张量维度是否正确
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
    
    // 提取基本参数
    const int P = means3D.size(0);  // 点的数量
    const int H = image_height;     // 图像高度
    const int W = image_width;      // 图像宽度

    // 创建张量选项，与输入张量保持一致的设备和数据类型
    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    // 初始化输出颜色张量，大小为[通道数, 高度, 宽度]，初始值为0
    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    // 初始化逆深度输出张量
    torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
    float* out_invdepthptr = nullptr;

    // 调整逆深度张量大小并获取数据指针
    out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
    out_invdepthptr = out_invdepth.data<float>();

    // 初始化每个点的半径张量
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
    
    // 创建CUDA设备上的缓冲区张量
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));      // 几何信息缓冲区
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));  // 分箱操作缓冲区
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));      // 图像数据缓冲区
    
    // 创建缓冲区大小调整函数
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
    
    int rendered = 0;  // 记录渲染的点数量
    // 如果有点需要渲染
    if(P != 0)
    {
        // 球谐函数系数数量
        int M = 0;
        if(sh.size(0) != 0)
        {
            M = sh.size(1);
        }

        // 调用CUDA光栅化器的前向渲染函数
        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,          // 几何缓冲区调整函数
            binningFunc,       // 分箱缓冲区调整函数
            imgFunc,           // 图像缓冲区调整函数
            P, degree, M,      // 点数量、球谐阶数、球谐系数数量
            background.contiguous().data<float>(),  // 背景颜色数据指针
            W, H,              // 图像宽高
            means3D.contiguous().data<float>(),     // 3D中心点数据指针
            sh.contiguous().data_ptr<float>(),      // 球谐系数数据指针
            colors.contiguous().data<float>(),      // 颜色数据指针
            opacity.contiguous().data<float>(),     // 不透明度数据指针
            scales.contiguous().data_ptr<float>(),  // 尺度参数数据指针
            scale_modifier,                          // 尺度修正因子
            rotations.contiguous().data_ptr<float>(),// 旋转参数数据指针
            cov3D_precomp.contiguous().data<float>(),// 预计算协方差矩阵数据指针
            viewmatrix.contiguous().data<float>(),  // 视图矩阵数据指针
            projmatrix.contiguous().data<float>(),  // 投影矩阵数据指针
            campos.contiguous().data<float>(),      // 相机位置数据指针
            tan_fovx, tan_fovy,                     // 视场角正切值
            prefiltered,                            // 是否预过滤
            out_color.contiguous().data<float>(),   // 输出颜色数据指针
            out_invdepthptr,                        // 输出逆深度数据指针
            antialiasing,                           // 是否抗锯齿
            radii.contiguous().data<int>(),         // 半径数据指针
            debug);                                 // 是否调试模式
    }
    
    // 返回渲染结果及相关缓冲区
    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}
```
输出结果是元组形式, 包括渲染点的数量, 输出颜色张量, 每个点的半径, 逆深度张量等.

主要实现流程是:
1. 输入验证（如检查means3D的维度是否为[N, 3]）。
2. 初始化输出张量（颜色、逆深度、半径等）和 `CUDA` 缓冲区。
3. 调用 `CUDA` 光栅化器的`forward`方法，执行核心渲染逻辑：
4. 将 3D 高斯体素通过视图矩阵和投影矩阵转换到 2D 图像平面。
5. 结合球谐函数计算光照，叠加颜色和不透明度，生成最终图像。
6. 返回渲染结果及中间缓冲区

其中, `CUDA`光栅化器的`forward`调用形式为:
```cpp
 rendered = CudaRasterizer::Rasterizer::forward()
```
这个方法在`submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer.h`中定义:
```cpp
namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

    //其他代码
    static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* depth,
			bool antialiasing,
			int* radii = nullptr,
			bool debug = false);
    //其他代码
    }
}
```

实现在`submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer_impl.cu`中:
```cpp
CudaRasterizer::Rasterizer::forward(
    //...
    )
    {
        //...
    }
```

#### 6.1.3 反向传播函数: `RasterizeGaussiansBackwardCUDA`
核心功能是计算损失函数对输入参数的梯度, 支持模型训练.

- 输入: 除正向渲染的参数外, 还包括输出颜色的梯度(`dL_dout_color`)和逆深度的梯度（`dL_dout_invdepth`）.
- 输出结果：各输入参数的梯度（如`dL_dmeans3D`为 3D 中心的梯度，`dL_dcolors`为颜色的梯度等）.
- 实现逻辑: 通过链式法则，从输出颜色的梯度反向推导各输入参数的梯度，依赖 `CUDA` 光栅化器的`backward`方法实现高效并行计算.

其中调用CUDA光栅化器反向传播的代码是:
```cpp
CudaRasterizer::Rasterizer::backward(
    //...
    )
    {
        //...
    }
```
在`submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer.h`中定义:
```cpp
namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
    //...
    static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_invdepths,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dinvdepth,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool antialiasing,
			bool debug);
	};
};
```
在`submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer_impl.cu`中实现:
```cpp
void CudaRasterizer::Rasterizer::backward(
	//...
    )
{
	//...
}
```

#### 6.1.4 可见性判断函数: `markVisible`
核心功能是判断3D高斯体素是否在当前相机视角下可见

- 输入参数: 3D 中心坐标（`means3D`）、视图矩阵（`viewmatrix`）、投影矩阵（`projmatrix`）.
- 输出结果: bool张量, True表示可见, False表示不可见.
- 实现逻辑: 通过视图矩阵和投影矩阵将 3D 点转换到裁剪空间，判断是否在视锥体范围内（未超出[-1, 1]范围）.

其中, 调用CUDA函数代码:
```cpp
// 调用CUDA函数标记可见点
        CudaRasterizer::Rasterizer::markVisible(
            P,                                      // 点数量
            means3D.contiguous().data<float>(),     // 3D中心点数据指针
            viewmatrix.contiguous().data<float>(),  // 视图矩阵数据指针
            projmatrix.contiguous().data<float>(),  // 投影矩阵数据指针
            present.contiguous().data<bool>());     // 可见性标记输出指针
```

其中, `markVisible`定义在文件`submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer.h`中:
```cpp
namespace CudaRasterizer
{
    class Rasterizer
    {
    public:
        //...
        static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present
		);
        //...
    }
}
```

该函数实现在`submodules\diff-gaussian-rasterization\cuda_rasterizer\rasterizer_impl.cu`中:
```cpp
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}
```
------