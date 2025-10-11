---
title: 'Blog Post number 4'
date: 2025-08-21
permalink: /posts/2025/08/blog-post-3/
tags:
  - PGSR
  - MoGe2
  - Python
  - C++
---

探索MoGe2和PGSR的浅层结合

# 使用MoGe2来监督PGSR的normal

MoGe2是一个单目深度估计网络，目前属于这个领域的SOTA方法。MoGe2可以渲染出一幅图像的深度信息以及平面法向量信息，而PGSR中的loss计算中也有这部分的内容，这启示我们可以使用MoGe2来监督PGSR的normal估计。

## 1. MoGe2网络的最简单使用方法
根据原项目地址中的`readme.md`，我们可以用下列代码来获得MoGe2网络的一个基本输出：
```py
import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)                             

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread("PATH_TO_IMAGE.jpg"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer 
output = model.infer(input_image)
"""
`output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
The maps are in the same size as the input image. 
{
    "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
    "depth": (H, W),        # depth map
    "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
    "mask": (H, W),         # a binary mask for valid pixels. 
    "intrinsics": (3, 3),   # normalized camera intrinsics
}
"""
```

这其中`output`是一个输出字典，其中包括了我们需要的normal信息，因此我们可以对这个脚本进行修改，使其更能够对应到PGSR的数据集组织方式。

我们需要对一个文件夹中的所有图片按顺序读取并获取对应的output，保存在另一个文件夹中，我将其命名为`normal_from_moge`。

具体脚本实现：
```py
import cv2
import argparse
import numpy as np
import torch
import os
from tqdm import tqdm
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

def main():
    parser = argparse.ArgumentParser(description='Run MoGe model with pretrained weights on all images in a folder.')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--image_folder', type=str, default='example_images', help='Path to the folder containing images.')
    args = parser.parse_args()

    device = torch.device("cuda")
    
    # Load the model from the specified path
    
    model = MoGeModel.from_pretrained(args.pretrained).to(device)
    
    # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
    
    # Get all image files in the folder
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    for root, _, files in os.walk(args.image_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    for image_path in tqdm(image_files, desc='Processing images'):
        image_id=os.path.basename(image_path).split('.')[0]
        try:
            input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
                    
            # Infer 
            output = model.infer(input_image)
            # print(f"Processed image: {image_path}")
                    
            # 获取 normal 数据
            if 'normal' in output:
                normal = output['normal']
                # 如果需要在 CPU 上使用 numpy 数组，可以添加以下代码
                normal = normal.cpu().numpy()
                # 修改形状从 (H, W, 3) 到 (3, H, W)
                normal = np.transpose(normal, (2, 0, 1))
                # print('成功获取 normal 数据，形状为:', normal.shape)

                # 保存 normal 数据
                parent_folder = os.path.dirname(args.image_folder)
                normal_folder = os.path.join(parent_folder, "normal_from_moge")
                os.makedirs(normal_folder, exist_ok=True)
                normal_path = os.path.join(normal_folder, f'{image_id}.png')
                # 转换形状为 (H, W, 3) 以适配 cv2.imwrite
                normal_to_save = np.transpose(normal, (1, 2, 0))
                cv2.imwrite(normal_path, (normal_to_save * 255).astype(np.uint8))
            else:
                print('output 中不包含 normal 数据。')
        except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
    """
    `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
    The maps are in the same size as the input image. 
    {
        "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
        "depth": (H, W),        # depth map
        "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
        "mask": (H, W),         # a binary mask for valid pixels. 
        "intrinsics": (3, 3),   # normalized camera intrinsics
    }
    """
    
    # # 获取 normal 数据
    # if 'normal' in output:
    #     normal = output['normal']
    #     # 如果需要在 CPU 上使用 numpy 数组，可以添加以下代码
    #     normal = normal.cpu().numpy()
    #     print('成功获取 normal 数据，形状为:', normal.shape)
    # else:
    #     print('output 中不包含 normal 数据。')

if __name__ == '__main__':
    main()
```

在获取每一个图像的normal后，将其保存为png格式，方便后续的PGSR读取。

## 2.PGSR与MoGe2的融合

在PGSR中，所有的loss计算均在`train.py`中实现。每一轮训练的流程如下：
1. 读取相机列表，随机选择一个相机（随机读取一个图像）
2. 获取其GT图像，GT灰度图
3. 执行`render`光栅化方法，获取模型渲染结果`render_pkg`，便于计算loss
4. 计算loss，包括 photometric loss, depth loss, normal loss
5. 反向传播，更新模型参数
6. 重复以上步骤，直到训练完成

为了在每一轮训练中，都能读取对应的normal_from_moge的图，我们需要知道当前进行训练的相机（图像）是哪一个，因此，对PGSR做出如下修改：

### 2.1 在获取相机的同时，获取图像编号

获取相机以及对应的编码，还有对应的normal_from_moge图像的代码如下：
```py
# Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray, image_path = viewpoint_cam.get_image()
        
        image_id=os.path.basename(image_path).split(".")[0]
        
        normal_image_from_moge=os.path.join((os.path.dirname(os.path.dirname(image_path))),"normal_from_moge",f"{image_id}.png")
```

在这里，`get_image()`返回的参数相较于原版PGSR多了一个`image_id`，用于获取对应的normal_from_moge图像。

这需要我们对`get_image()`方法进行修改。`get_image`实现在`scene/cameras.py`中：
```py
def get_image(self):
        if self.preload_img:
            return self.original_image.cuda(), self.original_image_gray.cuda(), self.image_path
        else:
            gt_image, gray_image, _ = process_image(self.image_path, self.resolution, self.ncc_scale)
            return gt_image.cuda(), gray_image.cuda(), self.image_path
```

只需要在`return`中返回一个图像路径，就可以在`train.py`中利用这个路径获取对应的图像编号。

获取图像编号后，找到PGSR中计算normal_loss的部分，为替换做准备。

### 2.2 替换normal_loss

PGSR中的`single-view loss`也就是单视角深度损失的计算方法如下：
```py
 # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]
            
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            
            if not opt.wo_image_weight:
                image_weight = erode(image_weight[None,None]).squeeze()
                
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()

            else:
                
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                
            loss += (normal_loss)
```

从`render`方法返回的`render_pkg`中获取`depth_normal`和`normal`，并利用`normal`监督`depth_normal`。因此，我们可以将其中的`normal`替换成由MoGe2获取的`normal_from_moge`。

流程如下：
1. 根据`image id`来读取对应的MoGe2渲染图像并转换格式：
```py
 # 读取图片
normal_from_moge = cv2.imread(normal_image_from_moge)
# OpenCV 默认读取为 BGR，转换为 RGB
normal_from_moge = cv2.cvtColor(normal_from_moge, cv2.COLOR_BGR2RGB)
# 转换为 3HW 形状
normal_from_moge = np.transpose(normal_from_moge, (2, 0, 1))
```

2. 将`numpy`数组转换为`Tensor`，并确保和`depth_normal`在相同设备上:
```py
 normal_from_moge = torch.from_numpy(normal_from_moge).to(depth_normal.device).to(depth_normal.dtype)
```

3. 替换`normal`为`normal_from_moge`，并计算`loss`：
```py
if not opt.wo_image_weight:
    image_weight = erode(image_weight[None,None]).squeeze()
                
    normal_loss = weight * (image_weight * (((normal_from_moge - depth_normal)).abs().sum(0))).mean()
else:
                
     normal_loss = weight * (((normal_from_moge - depth_normal)).abs().sum(0)).mean()
```


------