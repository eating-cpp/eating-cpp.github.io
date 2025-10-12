---
title: 'PGSR对比3DGS，我的一些学习与理解'
date: 2025-08-16
permalink: /posts/2025/08/blog-post-3/
tags:
  - 3DGS
  - PGSR
  - CUDA
  - Python
  - C++
---

3DGS与PGSR的对比与感悟。



# PGSR对比3DGS，我的一些学习与理解

PGSR在3DGS的基础上做了一些创新，达到了很好的效果，两者主要区别如下：
1. 3DGS 由于高斯点云的无结构和不规则性，仅依赖图像重建损失难以保证几何重建精度和多视图一致性，重建网格质量通常不尽如人意。PGSR 则通过**引入平面约束等多种方式**，实现了全局一致的几何重建，在几何重建精度上有显著提升。
2. 3DGS 没有专门针对深度渲染的优化方法，其深度相关计算可能与实际表面存在偏离。PGSR **提出了无偏深度渲染方法**，先渲染高斯平面到相机的距离图和法向图，再转换为无偏深度图，使渲染深度能与实际表面更好地一致。
3. 3DGS 未提及对光照变化的特殊处理。PGSR **提出了相机曝光补偿模型**，能够更好地应对场景中存在的大光照变化情况，进一步提升了重建精度。

下面就`train.py`开始谈谈我对PGSR的理解

## 1. `train.py`理解
`[PGSR] train.py`是在3DGS的`train.py`基础上做了一些修改, 主要代码构建, 流程都没有做过多变化, 只关注一些变化的地方.

### 1.1 初始化外观模型
在PGSR中, 训练开始前, 代码会对一个外观模型进行初始化, 这是3DGS所没有的, 具体如下:
```python
# 初始化外观模型
    app_model = AppModel()
    app_model.train()
    app_model.cuda()
```
该类实现在`scene/app_model.py`中:
```python
import torch
import torch.nn as nn
import os

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

class AppModel(nn.Module):
    def __init__(self, num_images=1600):  
        super().__init__()   
        self.appear_ab = nn.Parameter(torch.zeros(num_images, 2).cuda())
        self.optimizer = torch.optim.Adam([
                                {'params': self.appear_ab, 'lr': 0.001, "name": "appear_ab"},
                                ], betas=(0.9, 0.99))
            
    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "app_model/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        print(f"save app model. path: {out_weights_path}")
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'app.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "app_model"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "app_model/iteration_{}/app.pth".format(loaded_iter))
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)
```
定义了一个名为`AppModel`的类, 该类继承自`nn.Module`, 是一个pytorch类, 用于管理和操作模型的外观参数。 同时包含一个辅助函数`searchForMaxIteration`，用于查找指定文件夹中最大的迭代次数.

`_init_`是构造方法,其中有几个参数:
- `num_images`: 场景中图像的数量, 默认为1600
- self.appear_ab: 一个可训练的参数, 形状为`(num_images, 2)`, 用于存储每个图像的外观参数, 转移至cuda设备上
- self.optimizer: 一个优化器, 用于更新`self.appear_ab`, 学习率为0.001

`save_weights`方法用于保存模型的权重. 根据`model_path`和`iteration`参数, 构建出权重的保存路径, 并使用`torch.save`保存模型的状态字典至`app.pth`.

`load_weights`方法用于加载模型的权重. 根据`model_path`和`iteration`参数, 构建出权重的加载路径, 并使用`torch.load`加载模型的状态字典. 如果`iteration`为-1, 则会调用`searchForMaxIteration`函数查找最大的迭代次数.

回到我们的`train.py`。

### 1.2 各种损失的初始化
在3DGS中对损失的初始化只有：
```python
 # 用于日志记录的指数移动平均损失
    ema_loss_for_log = 0.0#训练过程中的总损失（loss），即包含所有损失项的加权和。
    
    
    # 用于日志记录的指数移动平均深度 L1 损失
    ema_Ll1depth_for_log = 0.0#仅针对深度 L1 损失（Ll1depth），即模型预测的深度与真实深度之间的 L1 误差（经权重调度后的值）。
```

而PGSR中对损失的初始化有：
```python
 # 用于日志记录的指数移动平均损失
    ema_loss_for_log = 0.0#训练过程中的总损失（loss），即包含所有损失项的加权和。
    
    # 用于日志记录的指数移动平均单视图损失
    ema_single_view_for_log = 0.0#仅针对单视图损失（single_view_loss），即模型在每个视图上的重建损失。
    
    # 用于日志记录的指数移动平均多视图几何损失
    ema_multi_view_geo_for_log = 0.0#仅针对多视图几何损失（multi_view_geo_loss），即模型在多个视图上的几何一致性损失。
    
    # 用于日志记录的指数移动平均多视图光度损失
    ema_multi_view_pho_for_log = 0.0#仅针对多视图光度损失（multi_view_pho_loss），即模型在多个视图上的光度一致性损失。

    normal_loss, geo_loss, ncc_loss = None, None, None  # 各种损失初始化
```

 ### 1.3 外观模型的启用
 PGSR规定在1000次训练后启用外观模型
```py
  # 迭代1000次后启用外观模型(如果设置了曝光补偿)
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True
```
其中`gaussians`是一个`Gaussian`类的实例化，这个类实现在`scene/gaussian_model.py`中。`gaussians.use_app()`也是原先3DGS没有的。具体代码如下：
```py
class Gaussianmodel:

    # ...
    def __init__(self, sh_degree : int):
            self.active_sh_degree = 0
            self.max_sh_degree = sh_degree  
            self._xyz = torch.empty(0)
            self._knn_f = torch.empty(0)
            self._features_dc = torch.empty(0)
            self._features_rest = torch.empty(0)
            self._scaling = torch.empty(0)
            self._rotation = torch.empty(0)
            self._opacity = torch.empty(0)
            self.max_radii2D = torch.empty(0)
            self.max_weight = torch.empty(0)
            self.xyz_gradient_accum = torch.empty(0)
            self.xyz_gradient_accum_abs = torch.empty(0)
            self.denom = torch.empty(0)
            self.denom_abs = torch.empty(0)
            self.optimizer = None
            self.percent_dense = 0
            self.spatial_lr_scale = 0
            self.knn_dists = None
            self.knn_idx = None
            self.setup_functions()
            self.use_app = False #默认为不使用外观模型
    #...
```

在这之前，还需要获取当前个相机的GT和灰度图，便于后续的外观模型训练。
```py
# 获取当前相机的Ground Truth图像和灰度图
        gt_image, gt_image_gray = viewpoint_cam.get_image()
```

### 1.4 当前视角的渲染
```py
# 渲染当前视角
        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            bg, 
            app_model=app_model,
            return_plane=iteration>opt.single_view_weight_from_iter, 
            return_depth_normal=iteration>opt.single_view_weight_from_iter
            )
```
`render`方法定义在`gaussian_renderer/__init__.py`中。

<details>
<summary>`gaussian_renderer/__init__.py`中的`render`方法</summary>

#### `gaussian_renderer/__init__.py`中的`render`方法
和3DGS的`render`方法类似，但在其上做了很多修改。

##### 1.4.1.1 参数部分
3DGS的`render`方法的参数如下：
```py
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
```

除了3DGS的参数，PGSR的`render`方法还添加了以下参数：
1. `app_model: AppModel=None`：外观模型对象，默认值为 `None`。如果提供了该参数，并且 `pc.use_app` 为 `True`，则会使用外观模型对渲染结果进行处理。
2. `return_plane = True`：是否返回平面信息，默认值为 `True`。如果为 `True`，则会在渲染结果中包含平面深度、法线等信息。
3. `return_depth_normal = True`：是否返回深度法线信息，默认值为 `True`。如果为 `True`，则会在渲染结果中包含深度法线信息。

##### 1.4.1.2 张量初始化部分
在3DGS中创建一个零张量，用来让Pytorch返回2D均值的梯度，在此外，PGSR还创建了一个零张量，用来让Pytorch返回2D法线的梯度。
```py
# 创建零张量用于存储屏幕空间的2D坐标，并启用梯度计算
    # 这些张量将用于获取2D均值的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 保留梯度以便反向传播时使用
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass
```

##### 1.4.1.3 光栅化器配置与创建
在PGSR中，光栅化器命名为`PlaneGaussianRasterizer`，对应的配置实例方法命名为`PlaneGaussianRasterizationSettings`：
```py
# 创建光栅化配置实例
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),  # 图像高度
            image_width=int(viewpoint_camera.image_width),    # 图像宽度
            tanfovx=tanfovx,                                  # 水平视场角正切值
            tanfovy=tanfovy,                                  # 垂直视场角正切值
            bg=bg_color,                                      # 背景颜色
            scale_modifier=scaling_modifier,                  # 缩放修正因子
            viewmatrix=viewpoint_camera.world_view_transform, # 视图变换矩阵（世界到相机）
            projmatrix=viewpoint_camera.full_proj_transform,  # 投影矩阵（相机到裁剪空间）
            sh_degree=pc.active_sh_degree,                    # 球谐函数的激活阶数
            campos=viewpoint_camera.camera_center,            # 相机位置
            prefiltered=False,                                # 是否预过滤（未使用）
            render_geo=return_plane,                          # 是否渲染几何信息（平面相关）
            debug=pipe.debug                                  # 是否启用调试模式
        )

    # 创建光栅化器实例
    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    # 若不返回平面信息，执行基础渲染流程
    if not return_plane:
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        
        # 构建返回字典，包含渲染结果和辅助信息
        return_dict = {
            "render": rendered_image,                     # 渲染的图像
            "viewspace_points": screenspace_points,       # 视空间中的2D点（带梯度）
            "viewspace_points_abs": screenspace_points_abs, # 视空间中的绝对2D点（带梯度）
            "visibility_filter": radii > 0,               # 可见性过滤（半径>0的高斯可见）
            "radii": radii,                               # 每个高斯在屏幕上的半径
            "out_observe": out_observe                    # 观测相关输出（具体依赖光栅化器实现）
        }
        # 若启用外观模型，计算外观调整后的图像并添加到返回字典
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict
```
这之中，`PlaneGaussianRasterizationSettings`和`PlaneGaussianRasterizer`定义在`ssubmodules/diff-plane-rasterization/diff_plane_rasterization/__init__.py`中。

这个package由`submodules/diff-plane-rasterization/setup.py`定义，将所有的扩展命名为`._C`。
<details>
<summary>代码</summary>

```py
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_plane_rasterization",
    packages=['diff_plane_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_plane_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

```
</details>
<br>



<details>
<summary>`diff_plane_rasterization/__init__.py`中的`PlaneGaussianRasterizer`类</summary>
<!-- details for 1.4.1.3 -->

其具体实现在`/home/lyj/anaconda3/envs/pgsr/lib/python3.8/site-packages/diff_plane_rasterization/__init__.py`中：

<details>
<summary>代码实现</summary>

```py
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, means2D_abs, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, all_map=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
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
        if all_map is None:
            all_map = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_abs,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            all_map,
            raster_settings, 
        )
```
</details>

GaussianRasterizer 继承自 torch.nn.Module，用于实现高斯点云的光栅化操作。该类封装了光栅化所需的设置和操作，提供了标记可见点和前向传播的功能。

1. `markVisible`：基于相机视锥体剔除，标记可见的点。
- 使用 torch.no_grad() 上下文管理器，避免计算梯度
- 调用 _C.mark_visible 函数，传入点的位置、视图矩阵和投影矩阵，得到可见点的布尔掩码。
<details>
<summary> markVisbile (CUDA) 代码实现 </summary>

`markVisible`由`submodules/diff-plane-rasterization/rasterize_points.cu`封装，其形式如下：
```cpp
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
```

具体实现在`submodules/diff-plane-rasterization/cuda_rasterizer/rasterizer_impl.cu`中：

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

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}
```

</details>

2. `forward`：前向传播函数，执行高斯点云的光栅化操作。
- 接收高斯点的位置、颜色、不透明度、缩放因子、旋转矩阵和协方差矩阵等输入。
- 调用 markVisible 函数标记可见点。
- 调用 rasterize_gaussians 函数执行光栅化操作，返回渲染结果、屏幕空间点、可见性过滤掩码、半径和观测相关输出。
- 若启用外观模型，计算外观调整后的图像并添加到返回字典。
`rasterize_gaussians`定义在同一个文件的一个方法中：
```py
def rasterize_gaussians(
    means3D,
    means2D,
    means2D_abs,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    all_map,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_map,
        raster_settings,
    )
```
`_RasterizeGaussians`定义在同一个文件中的一个类中,这个类包含了前向传播，反向传播的CUDA代码调用。
<details>
<summary>_RasterizeGaussians的定义和实现</summary>

```py
class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_maps,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            all_maps,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.render_geo,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, out_observe, out_all_map, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(out_all_map, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, out_observe, out_all_map, out_plane_depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_observe, grad_out_all_map, grad_out_plane_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        all_map_pixels, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                all_map_pixels,
                means3D, 
                radii, 
                colors_precomp, 
                all_maps,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_all_map,
                grad_out_plane_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.render_geo,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)
        # print(f"grad_means2D {grad_means2D.sum()}, grad_means2D_abs {grad_means2D_abs.sum()}")

        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_abs,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            gard_all_map,
            None,
        )

        return grads
```



</detials>
<br>

`rasterize_gaussians` 是一个独立函数，它接收一系列参数，然后直接调用 `_RasterizeGaussians.apply` 方法，并将传入的参数传递给该方法。`_RasterizeGaussians` 是 `torch.autograd.Function` 的子类，`apply` 方法用于调用其静态 `forward` 方法，`forward`方法中还会调用已经封装好的底层CUDA代码：
```py
# Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, out_observe, out_all_map, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
```
其中`_C.rasterize_gaussians`是一个CUDA函数，在 `ext.cpp` 文件中，使用 `pybind11` 将 `C++/CUDA` 函数绑定到 `Python`:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
```
真正实现在`rasterize_points.cu`文件中，`RasterizeGaussiansCUDA`函数调用了`rasterize_points`函数。

<details>
<summary>RasterizeGaussiansCUDA的调用链路</summary>

`rasterize_points.cu` 文件实现了 `RasterizeGaussiansCUDA` 函数调用：

```cpp
rendered = CudaRasterizer::Rasterizer::forward(
    //...
)
```

```cpp
void Rasterizer::forward(
    //...
)
{
    // 初始化渲染缓冲区
    initialize_render_buffers();

    // 执行点渲染
    render_points();

    // 执行三角形渲染
    render_triangles();

    // 合并渲染结果
    merge_render_results();
}
```

而这个`forward`函数中又被声明在`submodules/diff-plane-rasterization/cuda_rasterizer/rasterizer_impl.cu`中，`rasterizer_impl.cu`中的`forward`真正实现在`submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu`中。

同样的，`backward`也拥有类似的调用链轮：`loss.backward() -> _RasterizeGaussians.backward -> _C.rasterize_gaussians_backward`

</details>
<br>



<!-- remain to be implemented -->
</details>
<br>

##### 1.4.1.4 返回字典的构建

PGSR在构建返回字典是，还需要考虑平面信息和外观模型的使用与否。

<details>
<summary>返回字典的构建代码</summary>

```py
# 若不返回平面信息，执行基础渲染流程
    if not return_plane:
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
            )
        
        # 构建返回字典，包含渲染结果和辅助信息
        return_dict = {
            "render": rendered_image,                     # 渲染的图像
            "viewspace_points": screenspace_points,       # 视空间中的2D点（带梯度）
            "viewspace_points_abs": screenspace_points_abs, # 视空间中的绝对2D点（带梯度）
            "visibility_filter": radii > 0,               # 可见性过滤（半径>0的高斯可见）
            "radii": radii,                               # 每个高斯在屏幕上的半径
            "out_observe": out_observe                    # 观测相关输出（具体依赖光栅化器实现）
        }
        # 若启用外观模型，计算外观调整后的图像并添加到返回字典
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict

    # 若需要返回平面信息，计算额外的几何参数
    # 获取高斯在世界空间中的法向量，并转换到相机空间
    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3]
    # 计算高斯中心在相机空间中的坐标
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]
    depth_z = pts_in_cam[:, 2]  # 相机空间中的深度（z坐标）
    # 计算高斯中心到平面的距离（沿法向量方向）
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    # 构建包含法向量、alpha和距离的输入映射
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal  # 前3列为相机空间法向量
    input_all_map[:, 3] = 1.0            # 第4列为alpha值（固定为1）
    input_all_map[:, 4] = local_distance # 第5列为到平面的距离

    # 执行包含平面信息的光栅化
    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        means2D_abs=means2D_abs,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        all_map=input_all_map,       # 包含法向量、alpha和距离的映射
        cov3D_precomp=cov3D_precomp)

    # 从光栅化输出中解析法向量、alpha和距离
    rendered_normal = out_all_map[0:3]       # 渲染得到的法向量（3, H, W）
    rendered_alpha = out_all_map[3:4, ]      # 渲染得到的alpha通道（1, H, W）
    rendered_distance = out_all_map[4:5, ]   # 渲染得到的距离（1, H, W）
    
    # 构建包含平面信息的返回字典
    return_dict = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "viewspace_points_abs": screenspace_points_abs,
        "visibility_filter": radii > 0,
        "radii": radii,
        "out_observe": out_observe,
        "rendered_normal": rendered_normal,  # 渲染的法向量
        "plane_depth": plane_depth,          # 平面深度
        "rendered_distance": rendered_distance  # 渲染的距离
    }
    
    # 若启用外观模型，添加外观调整后的图像
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    # 若需要返回深度法向量，计算并添加到返回字典
    if return_depth_normal:
        # 从平面深度计算法向量，并乘以alpha通道（过滤背景）
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # 注：那些被视锥体剔除或半径为0的高斯不可见，将被排除在分割标准的更新之外
    return return_dict
```
</details>
<br>

整体逻辑描述：
1. 非平面渲染模式：
   - 当 `return_plane` 为 `False` 时，执行基础渲染流程，仅返回基本的渲染结果。
2. 平面渲染模式：
   - 当 `return_plane` 为 `True` 时，计算额外的几何参数，执行包含平面信息的光栅化，返回更丰富的渲染结果。
3. 外观模型处理：
   - 若启用外观模型且 `pc.use_app` 为 `True`，对渲染结果进行外观调整。
深度法线计算：若 `return_depth_normal` 为 `True`，计算并返回深度法线信息。

其中当 `return_plane` 为 `True` 时，计算高斯点在相机空间中的法向量、深度和到平面的距离，构建包含这些信息的输入映射 `input_all_map`

调用`rasterizer`进行包含平面信息的光栅化渲染，传入`input_all_map`获取结果，随后在从`out_all_map`中解析出得到的法向量、alpha通道和距离信息。


最后返回包含渲染结果、法向量、深度和距离等信息构成的字典。
<!-- details for 1.4.1 -->
</details>
<br>

</details>
<br>







### 1.5 损失的计算
在PGSR中，损失计算相对于3DGS而言更多。
除了基本的图像基础损失以及ssim，PGSR额外引入了尺度损失、单视角损失、多视角损失。


<details>
<summary>损失计算的代码</summary>

```py
 # 计算基础图像损失
        ssim_loss = (1.0 - ssim(image, gt_image))  # SSIM损失
        # 如果使用外观模型且SSIM损失足够小，使用外观模型输出计算L1损失
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)  # L1损失
            
        # 总图像损失：L1和SSIM的加权和
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()  # 初始化总损失
        
        # 尺度损失：约束高斯分布的尺度，防止过小
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]  # 取最小尺度
            loss += opt.scale_loss_weight * min_scale_loss.mean()
        
        # 单视角损失：法线一致性损失
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]  # 渲染得到的法线
            depth_normal = render_pkg["depth_normal"]  # 从深度图计算的法线

            # 图像权重：根据图像梯度调整权重，边缘区域权重低
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # 应用图像权重到法线损失
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)

        # 多视角损失：几何一致性和光度一致性损失
        if iteration > opt.multi_view_weight_from_iter:
            # 选择一个邻近相机
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else \
                scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            # 有一定概率使用虚拟相机
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, 
                                           deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            
            if nearest_cam is not None:
                # 多视角损失参数
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2  #  patch总像素数
                ncc_weight = opt.multi_view_ncc_weight  # 光度损失权重
                geo_weight = opt.multi_view_geo_weight  # 几何损失权重
                
                # 计算几何一致性掩码和损失
                H, W = render_pkg['plane_depth'].squeeze().shape
                # 生成像素坐标网格
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                # 渲染邻近相机视角
                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                # 从深度图获取3D点，并转换到邻近相机坐标系
                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                # 获取这些点在邻近相机深度图中的深度
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                # 投影一致性检查
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                # 投影回原视角图像平面
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                # 计算投影误差
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                
                # 根据投影误差和深度掩码计算权重
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                
                # 每200次迭代保存调试图像
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                # 计算几何损失
                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    
                    # 如果不是虚拟相机，计算光度一致性损失
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            # 采样有效像素
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            # 生成参考图像的patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            # 归一化到[-1,1]范围，用于grid_sample
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            # 采样参考图像的patch
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            # 计算参考相机到邻近相机的变换
                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        # 计算单应性矩阵(Homography)
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        
                        # 构建单应性矩阵
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        # 计算邻近图像的patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        # 采样邻近图像的patch
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        # 计算NCC(归一化互相关)损失
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

```
</details>
<br>


### 1.6 高斯点增密和剪枝
与3DGS的流程基本相同，主要的区别在于`add_densification_stats`和`densify_and_prune`的参数传递中。

1. `add_densification_stats`
具体实现在`scene/gaussian_model.py`

<details>
<summary>代码实现</summary>

```py
def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1
```
</details>
<br>
该方法通过累加梯度信息和更新计数，为后续的高斯点云加密和修剪操作提供统计依据。通过这些统计信息，可以判断哪些区域的点需要加密以提高模型精度，哪些点需要修剪以减少计算量



2. `densify_and_prune`
具体实现在`scene/gaussian_model.py`中：

<details>
<summary>代码实现</summary>

```py
def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()
```

</details>
<br>

主要用于对高斯点进行加密和修剪操作，以优化高斯点云的质量。具体步骤如下：

1. 梯度计算：计算平均梯度并处理其中的 NaN 值。
2. 点云加密：调用 densify_and_clone 和 densify_and_split 方法对高斯点进行加密。
3. 点云修剪：根据不透明度、二维半径和缩放比例创建修剪掩码，移除不符合条件的点。
4. 内存管理：清空 CUDA 缓存，释放 GPU 内存。

### 1.7 优化器更新
PGSR的优化器更新与3DGS基本相同，PSGR还额外更新了一个模型。

```py
     # 优化器更新
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)
```

------